/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseTool,
  ToolResult,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolMcpConfirmationDetails,
  Icon,
} from './tools.js';
import {
  CallableTool,
  Part,
  FunctionCall,
  FunctionDeclaration,
  Type,
} from '@google/genai';

type ToolParams = Record<string, unknown>;

export class DiscoveredMCPTool extends BaseTool<ToolParams, ToolResult> {
  private static readonly allowlist: Set<string> = new Set();

  constructor(
    private readonly mcpTool: CallableTool,
    readonly serverName: string,
    readonly serverToolName: string,
    description: string,
    readonly parameterSchemaJson: unknown,
    readonly timeout?: number,
    readonly trust?: boolean,
    nameOverride?: string,
  ) {
    super(
      nameOverride ?? generateValidName(serverToolName),
      `${serverToolName} (${serverName} MCP Server)`,
      description,
      Icon.Hammer,
      { type: Type.OBJECT }, // this is a dummy Schema for MCP, will be not be used to construct the FunctionDeclaration
      true, // isOutputMarkdown
      false, // canUpdateOutput
    );
  }

  asFullyQualifiedTool(): DiscoveredMCPTool {
    return new DiscoveredMCPTool(
      this.mcpTool,
      this.serverName,
      this.serverToolName,
      this.description,
      this.parameterSchemaJson,
      this.timeout,
      this.trust,
      `${this.serverName}__${this.serverToolName}`,
    );
  }

  /**
   * Overrides the base schema to use parametersJsonSchema when building
   * FunctionDeclaration
   */
  override get schema(): FunctionDeclaration {
    return {
      name: this.name,
      description: this.description,
      parametersJsonSchema: this.parameterSchemaJson,
    };
  }

  async shouldConfirmExecute(
    _params: ToolParams,
    _abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    const serverAllowListKey = this.serverName;
    const toolAllowListKey = `${this.serverName}.${this.serverToolName}`;

    if (this.trust) {
      return false; // server is trusted, no confirmation needed
    }

    if (
      DiscoveredMCPTool.allowlist.has(serverAllowListKey) ||
      DiscoveredMCPTool.allowlist.has(toolAllowListKey)
    ) {
      return false; // server and/or tool already allowlisted
    }

    const confirmationDetails: ToolMcpConfirmationDetails = {
      type: 'mcp',
      title: 'Confirm MCP Tool Execution',
      serverName: this.serverName,
      toolName: this.serverToolName, // Display original tool name in confirmation
      toolDisplayName: this.name, // Display global registry name exposed to model and user
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlwaysServer) {
          DiscoveredMCPTool.allowlist.add(serverAllowListKey);
        } else if (outcome === ToolConfirmationOutcome.ProceedAlwaysTool) {
          DiscoveredMCPTool.allowlist.add(toolAllowListKey);
        }
      },
    };
    return confirmationDetails;
  }

  async execute(params: ToolParams): Promise<ToolResult> {
    const functionCalls: FunctionCall[] = [
      {
        name: this.serverToolName,
        args: params,
      },
    ];

    const rawResponseParts = await this.mcpTool.callTool(functionCalls);
    const responseParts = normalizeResponseParts(rawResponseParts);

    // Check if the tool returned an error.
    const errorPart = responseParts.find(
      (p) => (p as { isError?: boolean }).isError,
    );

    let llmContent: Part[];

    if (errorPart) {
      const errorMessage =
        errorPart.text ?? JSON.stringify(errorPart, null, 2);
      llmContent = [
        {
          text: `The tool execution for ${this.serverToolName} failed with the following error: ${errorMessage}`,
        },
      ];
    } else {
      // Add a source message for the model to provide context.
      llmContent = [
        {
          text: `Response from tool ${this.serverToolName} from ${this.serverName} MCP Server:`,
        },
        ...responseParts,
      ];
    }

    return {
      llmContent: llmContent,
      returnDisplay: getStringifiedResultForDisplay(responseParts),
    };
  }
}

/**
 * A helper function to handle various formats of tool responses and normalize
 * them into the standard `Part[]` format that the SDK expects.
 */
function normalizeResponseParts(
  rawParts: Part[] | null | undefined,
): Part[] {
  if (!rawParts) {
    return [];
  }

  const unwrappedParts = unwrapFunctionResponse(rawParts);
  const deserializedItems = deserializeParts(unwrappedParts);
  const sdkParts = transformPartsToSdkFormat(deserializedItems);

  return sdkParts;
}

/**
 * Unwraps the `functionResponse` wrapper from the SDK if it exists.
 * @param parts The initial parts array from the SDK.
 * @returns The nested `content` if it exists, otherwise the original parts.
 */
function unwrapFunctionResponse(parts: Part[]): Part[] {
  if (
    parts.length === 1 &&
    parts[0].functionResponse &&
    typeof parts[0].functionResponse.response === 'object' &&
    parts[0].functionResponse.response !== null &&
    'content' in parts[0].functionResponse.response
  ) {
    return (parts[0].functionResponse.response as { content: Part[] }).content;
  }
  return parts;
}

/**
 * Deserializes parts that may be in a stringified JSON format.
 * @param parts The array of parts to process.
 * @returns An array of parsed, unknown objects.
 */
function deserializeParts(parts: Part[]): unknown[] {
  // Case A: The response is a single text part containing a stringified
  // JSON array of parts (legacy format).
  if (parts.length === 1 && parts[0].text) {
    try {
      const parsed = JSON.parse(parts[0].text);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch (_e) {
      // It's just a regular text part, not JSON.
      return parts;
    }
  }

  // Case B: The response is an array of text parts, each containing
  // a stringified JSON object (legacy format).
  if (parts.every((p) => p && p.text)) {
    return parts.map((p) => {
      try {
        return JSON.parse(p.text!);
      } catch (_e) {
        // If parsing fails, treat it as a regular text part.
        return p;
      }
    });
  }

  // Case C: It's already an array of Part-like objects.
  return parts;
}

// Helper types and type guards for robust type checking.
type McpPart = {
  type: 'image' | 'audio' | 'video' | 'text';
  data?: string;
  mimeType?: string;
  text?: string;
};

function isMcpMediaPart(item: any): item is McpPart {
  return (
    item &&
    (item.type === 'image' || item.type === 'audio' || item.type === 'video') &&
    typeof item.data === 'string' &&
    typeof item.mimeType === 'string'
  );
}

function isMcpTextPart(item: any): item is McpPart {
  return item && item.type === 'text' && typeof item.text === 'string';
}

function isSdkPart(item: any): item is Part {
  return (
    item &&
    ('text' in item || 'inlineData' in item || 'functionCall' in item)
  );
}

/**
 * Transforms an array of parsed objects into the Gemini SDK `Part` format.
 * @param items The array of unknown items to transform.
 * @returns A clean array of SDK `Part` objects.
 */
function transformPartsToSdkFormat(items: unknown[]): Part[] {
  return items
    .map((item: unknown): Part | null => {
      if (!item || typeof item !== 'object') {
        return null;
      }

      // Handle MCP-spec objects that need transformation FIRST.
      if (isMcpMediaPart(item)) {
        return {
          inlineData: {
            data: item.data!,
            mimeType: item.mimeType!,
          },
        };
      }

      if (isMcpTextPart(item)) {
        return {
          text: item.text!,
        };
      }

      // If it's not a transformable MCP part, check if it's already a valid SDK Part.
      if (isSdkPart(item)) {
        return item;
      }

      // Pass through any other unknown format.
      return item as Part;
    })
    .filter((p): p is Part => p !== null);
}

/**
 * Processes an array of `Part` objects from a tool's execution result
 * to generate a user-friendly string representation for display in a CLI.
 *
 * - `text` parts are concatenated.
 * - `inlineData` parts (like images) are represented by a placeholder.
 * - Other parts are stringified as JSON.
 *
 * @param result The array of `Part` objects to process.
 * @returns A string formatted for display.
 */
function getStringifiedResultForDisplay(result: Part[]): string {
  if (!result || result.length === 0) {
    return 'Tool returned no output.';
  }

  const displayParts: string[] = [];

  for (const part of result) {
    const toolError = part as { isError?: boolean; text?: string };
    // The SDK doesn't have isError on the type, so we cast to a safe type.
    if (toolError.isError) {
      displayParts.push(`Tool Error: ${toolError.text ?? ''}`);
      continue;
    }

    if (part.text) {
      displayParts.push(part.text);
    } else if (part.inlineData) {
      const mimeType = part.inlineData.mimeType ?? 'unknown';
      const type = mimeType.startsWith('audio/')
        ? 'Audio'
        : mimeType.startsWith('image/')
          ? 'Image'
          : mimeType.startsWith('video/')
            ? 'Video'
            : 'File';
      displayParts.push(`[${type} Content: ${mimeType}]`);
    } else {
      displayParts.push('```json\n' + JSON.stringify(part, null, 2) + '\n```');
    }
  }

  return displayParts.join('\n');
}

/** Visible for testing */
export function generateValidName(name: string) {
  // Replace invalid characters (based on 400 error message from Gemini API) with underscores
  let validToolname = name.replace(/[^a-zA-Z0-9_.-]/g, '_');

  // If longer than 63 characters, replace middle with '___'
  // (Gemini API says max length 64, but actual limit seems to be 63)
  if (validToolname.length > 63) {
    validToolname =
      validToolname.slice(0, 28) + '___' + validToolname.slice(-32);
  }
  return validToolname;
}