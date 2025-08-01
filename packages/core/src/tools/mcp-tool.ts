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

    // Add a source message for the model to provide context.
    const responseWithSource: Part[] = [
      {
        text: `Response from tool ${this.serverToolName} from ${this.serverName} MCP Server:`,
      },
      ...responseParts,
    ];

    return {
      llmContent: responseWithSource,
      returnDisplay: getStringifiedResultForDisplay(responseParts),
    };
  }
}

/**
 * A helper function to handle various formats of tool responses and normalize
 * them into the standard `Part[]` format that the SDK expects.
 */
function normalizeResponseParts(rawParts: Part[]): Part[] {
  // Start by filtering out any null/undefined entries.
  const validParts = rawParts.filter((p): p is Part => !!p);

  let itemsToProcess: unknown[];

  // Case 1: The response is a single text part containing a stringified
  // JSON array of parts.
  if (validParts.length === 1 && validParts[0].text) {
    try {
      const parsed = JSON.parse(validParts[0].text);
      itemsToProcess = Array.isArray(parsed) ? parsed : [parsed];
    } catch (_e) {
      // It's just a regular text part, not JSON.
      return validParts;
    }
  } else if (validParts.every((p) => p.text)) {
    // Case 2: The response is an array of text parts, each containing
    // a stringified JSON object.
    itemsToProcess = validParts.map((p) => {
      try {
        return JSON.parse(p.text!);
      } catch (_e) {
        return p;
      }
    });
  } else {
    // It's already an array of objects.
    itemsToProcess = validParts;
  }

  // Transform each item from the MCP spec format to the SDK format.
  return itemsToProcess
    .map((item: unknown): Part | null => {
      if (!item || typeof item !== 'object') {
        return null;
      }

      // Case B: An MCP-spec object that needs transformation.
      const mcpPart = item as {
        type?: string;
        data?: string;
        mimeType?: string;
        text?: string;
      };

      if (
        (mcpPart.type === 'image' ||
          mcpPart.type === 'audio' ||
          mcpPart.type === 'video') &&
        mcpPart.data &&
        mcpPart.mimeType
      ) {
        return {
          inlineData: {
            data: mcpPart.data,
            mimeType: mcpPart.mimeType,
          },
        };
      }

      if (mcpPart.type === 'text' && typeof mcpPart.text === 'string') {
        return {
          text: mcpPart.text,
        };
      }

      // Case A or C: Already a valid SDK Part or some other object.
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
