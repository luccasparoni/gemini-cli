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

/**
 * Transforms the raw MCP content blocks from the SDK response into a
 * standard GenAI Part array.
 * @param sdkResponse The raw Part[] array from `mcpTool.callTool()`.
 * @returns A clean Part[] array ready for the scheduler.
 */
function transformMcpContentToParts(sdkResponse: Part[]): Part[] {
  const funcResponse = sdkResponse?.[0]?.functionResponse;
  const mcpContent = funcResponse?.response?.content as Array<
    Record<string, string | Record<string, string>>
  >;
  const toolName = funcResponse?.name || 'unknown tool';

  if (!Array.isArray(mcpContent)) {
    return [{ text: '[Error: Could not parse tool response]' }];
  }

  const transformed = mcpContent.flatMap(
    (
      block: Record<string, string | Record<string, string>>,
    ): Part | Part[] | null => {
      let actualBlock = block;
      if (block.text && !block.type) {
        try {
          actualBlock = JSON.parse(block.text as string);
        } catch (_e) {
          return { text: block.text as string };
        }
      }
      switch (actualBlock.type) {
        case 'text':
          return { text: actualBlock.text as string };

        case 'image':
        case 'audio':
          return [
            {
              text: `[Tool '${toolName}' provided the following ${
                actualBlock.type
              } data with mime-type: ${actualBlock.mimeType as string}]`,
            },
            {
              inlineData: {
                mimeType: actualBlock.mimeType as string,
                data: actualBlock.data as string,
              },
            },
          ];

        case 'resource':
          if ((actualBlock.resource as Record<string, string>)?.text) {
            return {
              text: (actualBlock.resource as Record<string, string>)
                .text as string,
            };
          }
          if ((actualBlock.resource as Record<string, string>)?.blob) {
            return [
              {
                text: `[Tool '${toolName}' provided the following embedded resource with mime-type: ${
                  ((actualBlock.resource as Record<string, string>)
                    .mimeType as string) || 'application/octet-stream'
                }]`,
              },
              {
                inlineData: {
                  mimeType:
                    ((actualBlock.resource as Record<string, string>)
                      .mimeType as string) || 'application/octet-stream',
                  data: (actualBlock.resource as Record<string, string>)
                    .blob as string,
                },
              },
            ];
          }
          return null;

        case 'resource_link':
          return {
            text: `Resource Link: ${
              (actualBlock.title as string) || (actualBlock.name as string)
            } at ${actualBlock.uri as string}`,
          };

        default:
          return null;
      }
    },
  );

  return transformed.filter((part): part is Part => part !== null);
}

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
    const transformedParts = transformMcpContentToParts(rawResponseParts);

    return {
      llmContent: transformedParts,
      returnDisplay: getStringifiedResultForDisplay(rawResponseParts),
    };
  }
}

/**
 * Processes the raw response from the MCP tool to generate a clean,
 * human-readable string for display in the CLI. It summarizes non-text
 * content and presents text directly.
 *
 * @param rawResponse The raw Part[] array from the GenAI SDK.
 * @returns A formatted string representing the tool's output.
 */
function getStringifiedResultForDisplay(rawResponse: Part[]): string {
  // Safely extract the MCP content array.
  const mcpContent = rawResponse?.[0]?.functionResponse?.response
    ?.content as Array<Record<string, string | Record<string, string>>>;

  if (!Array.isArray(mcpContent)) {
    // Fallback for unexpected structures: pretty-print the raw response.
    return '```json\n' + JSON.stringify(rawResponse, null, 2) + '\n```';
  }

  const displayParts = mcpContent.map(
    (block: Record<string, string | Record<string, string>>): string => {
      switch (block.type) {
        case 'text':
          return block.text as string;
        case 'image':
          return `[Image: ${block.mimeType as string}]`;
        case 'audio':
          return `[Audio: ${block.mimeType as string}]`;
        case 'resource_link':
          return `[Link to ${
            (block.title as string) || (block.name as string)
          }: ${block.uri as string}]`;
        case 'resource':
          if ((block.resource as Record<string, string>)?.text) {
            return (block.resource as Record<string, string>).text as string;
          }
          return `[Embedded Resource: ${
            ((block.resource as Record<string, string>)?.mimeType as string) ||
            'unknown type'
          }]`;
        default:
          return `[Unknown content type: ${block.type as string}]`;
      }
    },
  );

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
