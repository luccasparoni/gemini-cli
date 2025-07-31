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
 * A helper function to handle servers that incorrectly serialize a Part[]
 * into a JSON string and return it as a single text part.
 */
function normalizeResponseParts(parts: Part[]): Part[] {
  // We only care about the case where the server sent a single text part.
  if (parts.length === 1 && parts[0].text) {
    try {
      const parsed = JSON.parse(parts[0].text);

      // We only act if the parsed result is a non-null object (which includes arrays).
      // If it's a string, number, or boolean, we treat it as literal text and do nothing.
      if (typeof parsed === 'object' && parsed !== null) {
        // If it's an array, we assume it's the Part array we want.
        if (Array.isArray(parsed)) {
          return parsed as Part[];
        }

        // If it's a single object that looks like a Part, wrap it in an array.
        if ('inlineData' in parsed || 'text' in parsed) {
          return [parsed as Part];
        }
      }
    } catch (e) {
      // The text was not valid JSON. It's just plain text.
      // We do nothing and fall through to return the original parts.
    }
  }

  // If any of the conditions fail, return the original parts unchanged.
  return parts;
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
    if (part.text) {
      displayParts.push(part.text);
    } else if (part.inlineData) {
      displayParts.push(`[Image Content: ${part.inlineData.mimeType}]`);
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
