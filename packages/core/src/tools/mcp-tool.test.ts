/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
  Mocked,
} from 'vitest';
import { DiscoveredMCPTool, generateValidName } from './mcp-tool.js'; // Added getStringifiedResultForDisplay
import { ToolResult, ToolConfirmationOutcome } from './tools.js'; // Added ToolConfirmationOutcome
import { CallableTool, Part } from '@google/genai';

// Mock @google/genai mcpToTool and CallableTool
// We only need to mock the parts of CallableTool that DiscoveredMCPTool uses.
const mockCallTool = vi.fn();
const mockToolMethod = vi.fn();

const mockCallableToolInstance: Mocked<CallableTool> = {
  tool: mockToolMethod as any, // Not directly used by DiscoveredMCPTool instance methods
  callTool: mockCallTool as any,
  // Add other methods if DiscoveredMCPTool starts using them
};

describe('generateValidName', () => {
  it('should return a valid name for a simple function', () => {
    expect(generateValidName('myFunction')).toBe('myFunction');
  });

  it('should replace invalid characters with underscores', () => {
    expect(generateValidName('invalid-name with spaces')).toBe(
      'invalid-name_with_spaces',
    );
  });

  it('should truncate long names', () => {
    expect(generateValidName('x'.repeat(80))).toBe(
      'xxxxxxxxxxxxxxxxxxxxxxxxxxxx___xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    );
  });

  it('should handle names with only invalid characters', () => {
    expect(generateValidName('!@#$%^&*()')).toBe('__________');
  });

  it('should handle names that are exactly 63 characters long', () => {
    expect(generateValidName('a'.repeat(63)).length).toBe(63);
  });

  it('should handle names that are exactly 64 characters long', () => {
    expect(generateValidName('a'.repeat(64)).length).toBe(63);
  });

  it('should handle names that are longer than 64 characters', () => {
    expect(generateValidName('a'.repeat(80)).length).toBe(63);
  });
});

describe('DiscoveredMCPTool', () => {
  const serverName = 'mock-mcp-server';
  const serverToolName = 'actual-server-tool-name';
  const baseDescription = 'A test MCP tool.';
  const inputSchema: Record<string, unknown> = {
    type: 'object' as const,
    properties: { param: { type: 'string' } },
    required: ['param'],
  };

  beforeEach(() => {
    mockCallTool.mockClear();
    mockToolMethod.mockClear();
    // Clear allowlist before each relevant test, especially for shouldConfirmExecute
    (DiscoveredMCPTool as any).allowlist.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should set properties correctly', () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      expect(tool.name).toBe(serverToolName);
      expect(tool.schema.name).toBe(serverToolName);
      expect(tool.schema.description).toBe(baseDescription);
      expect(tool.schema.parameters).toBeUndefined();
      expect(tool.schema.parametersJsonSchema).toEqual(inputSchema);
      expect(tool.serverToolName).toBe(serverToolName);
      expect(tool.timeout).toBeUndefined();
    });

    it('should accept and store a custom timeout', () => {
      const customTimeout = 5000;
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
        customTimeout,
      );
      expect(tool.timeout).toBe(customTimeout);
    });
  });

  describe('execute', () => {
    it('should call mcpTool.callTool with correct parameters and format display output', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const params = { param: 'testValue' };
      const mockToolSuccessResultObject = {
        success: true,
        details: 'executed',
      };
      const mockFunctionResponseContent: Part[] = [
        { text: JSON.stringify(mockToolSuccessResultObject) },
      ];
      const mockMcpToolResponseParts: Part[] = [
        {
          functionResponse: {
            name: serverToolName,
            response: { content: mockFunctionResponseContent },
          },
        },
      ];
      mockCallTool.mockResolvedValue(mockMcpToolResponseParts);

      const toolResult: ToolResult = await tool.execute(params);

      expect(mockCallTool).toHaveBeenCalledWith([
        { name: serverToolName, args: params },
      ]);
      expect(toolResult.llmContent).toEqual(mockMcpToolResponseParts);

      const stringifiedResponseContent = JSON.stringify(
        mockToolSuccessResultObject,
      );
      expect(toolResult.returnDisplay).toBe(stringifiedResponseContent);
    });

    it('should handle empty result from getStringifiedResultForDisplay', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const params = { param: 'testValue' };
      const mockMcpToolResponsePartsEmpty: Part[] = [];
      mockCallTool.mockResolvedValue(mockMcpToolResponsePartsEmpty);
      const toolResult: ToolResult = await tool.execute(params);
      expect(toolResult.returnDisplay).toBe('```json\n[]\n```');
    });

    it('should propagate rejection if mcpTool.callTool rejects', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const params = { param: 'failCase' };
      const expectedError = new Error('MCP call failed');
      mockCallTool.mockRejectedValue(expectedError);

      await expect(tool.execute(params)).rejects.toThrow(expectedError);
    });
  });

  describe('shouldConfirmExecute', () => {
    // beforeEach is already clearing allowlist

    it('should return false if trust is true', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
        undefined,
        true,
      );
      expect(
        await tool.shouldConfirmExecute({}, new AbortController().signal),
      ).toBe(false);
    });

    it('should return false if server is allowlisted', async () => {
      (DiscoveredMCPTool as any).allowlist.add(serverName);
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      expect(
        await tool.shouldConfirmExecute({}, new AbortController().signal),
      ).toBe(false);
    });

    it('should return false if tool is allowlisted', async () => {
      const toolAllowlistKey = `${serverName}.${serverToolName}`;
      (DiscoveredMCPTool as any).allowlist.add(toolAllowlistKey);
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      expect(
        await tool.shouldConfirmExecute({}, new AbortController().signal),
      ).toBe(false);
    });

    it('should return confirmation details if not trusted and not allowlisted', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const confirmation = await tool.shouldConfirmExecute(
        {},
        new AbortController().signal,
      );
      expect(confirmation).not.toBe(false);
      if (confirmation && confirmation.type === 'mcp') {
        // Type guard for ToolMcpConfirmationDetails
        expect(confirmation.type).toBe('mcp');
        expect(confirmation.serverName).toBe(serverName);
        expect(confirmation.toolName).toBe(serverToolName);
      } else if (confirmation) {
        // Handle other possible confirmation types if necessary, or strengthen test if only MCP is expected
        throw new Error(
          'Confirmation was not of expected type MCP or was false',
        );
      } else {
        throw new Error(
          'Confirmation details not in expected format or was false',
        );
      }
    });

    it('should add server to allowlist on ProceedAlwaysServer', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const confirmation = await tool.shouldConfirmExecute(
        {},
        new AbortController().signal,
      );
      expect(confirmation).not.toBe(false);
      if (
        confirmation &&
        typeof confirmation === 'object' &&
        'onConfirm' in confirmation &&
        typeof confirmation.onConfirm === 'function'
      ) {
        await confirmation.onConfirm(
          ToolConfirmationOutcome.ProceedAlwaysServer,
        );
        expect((DiscoveredMCPTool as any).allowlist.has(serverName)).toBe(true);
      } else {
        throw new Error(
          'Confirmation details or onConfirm not in expected format',
        );
      }
    });

    it('should add tool to allowlist on ProceedAlwaysTool', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const toolAllowlistKey = `${serverName}.${serverToolName}`;
      const confirmation = await tool.shouldConfirmExecute(
        {},
        new AbortController().signal,
      );
      expect(confirmation).not.toBe(false);
      if (
        confirmation &&
        typeof confirmation === 'object' &&
        'onConfirm' in confirmation &&
        typeof confirmation.onConfirm === 'function'
      ) {
        await confirmation.onConfirm(ToolConfirmationOutcome.ProceedAlwaysTool);
        expect((DiscoveredMCPTool as any).allowlist.has(toolAllowlistKey)).toBe(
          true,
        );
      } else {
        throw new Error(
          'Confirmation details or onConfirm not in expected format',
        );
      }
    });

    it('should handle Cancel confirmation outcome', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const confirmation = await tool.shouldConfirmExecute(
        {},
        new AbortController().signal,
      );
      expect(confirmation).not.toBe(false);
      if (
        confirmation &&
        typeof confirmation === 'object' &&
        'onConfirm' in confirmation &&
        typeof confirmation.onConfirm === 'function'
      ) {
        // Cancel should not add anything to allowlist
        await confirmation.onConfirm(ToolConfirmationOutcome.Cancel);
        expect((DiscoveredMCPTool as any).allowlist.has(serverName)).toBe(
          false,
        );
        expect(
          (DiscoveredMCPTool as any).allowlist.has(
            `${serverName}.${serverToolName}`,
          ),
        ).toBe(false);
      } else {
        throw new Error(
          'Confirmation details or onConfirm not in expected format',
        );
      }
    });

    it('should handle ProceedOnce confirmation outcome', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );
      const confirmation = await tool.shouldConfirmExecute(
        {},
        new AbortController().signal,
      );
      expect(confirmation).not.toBe(false);
      if (
        confirmation &&
        typeof confirmation === 'object' &&
        'onConfirm' in confirmation &&
        typeof confirmation.onConfirm === 'function'
      ) {
        // ProceedOnce should not add anything to allowlist
        await confirmation.onConfirm(ToolConfirmationOutcome.ProceedOnce);
        expect((DiscoveredMCPTool as any).allowlist.has(serverName)).toBe(
          false,
        );
        expect(
          (DiscoveredMCPTool as any).allowlist.has(
            `${serverName}.${serverToolName}`,
          ),
        ).toBe(false);
      } else {
        throw new Error(
          'Confirmation details or onConfirm not in expected format',
        );
      }
    });
  });

  describe('Media Content Transformation', () => {
    it('should transform single image content to inlineData', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const imageContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [{
              type: 'image',
              data: 'base64ImageData',
              mimeType: 'image/png',
            }],
          },
        },
      }];

      mockCallTool.mockResolvedValue(imageContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        {
          inlineData: {
            data: 'base64ImageData',
            mimeType: 'image/png',
          },
        },
      ]);
    });

    it('should handle mixed text and image content', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const mixedContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'text', text: 'Here is your image:' },
              { type: 'image', data: 'base64ImageData', mimeType: 'image/jpeg' },
              { type: 'text', text: 'Analysis complete.' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(mixedContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        { text: 'Here is your image:' },
        { inlineData: { data: 'base64ImageData', mimeType: 'image/jpeg' } },
        { text: 'Analysis complete.' },
      ]);
    });

    it('should handle multiple media items', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const multipleMedia: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'image', data: 'image1Data', mimeType: 'image/png' },
              { type: 'image', data: 'image2Data', mimeType: 'image/jpeg' },
              { type: 'pdf', data: 'pdfData', mimeType: 'application/pdf' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(multipleMedia);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        { inlineData: { data: 'image1Data', mimeType: 'image/png' } },
        { inlineData: { data: 'image2Data', mimeType: 'image/jpeg' } },
        { inlineData: { data: 'pdfData', mimeType: 'application/pdf' } },
      ]);
    });

    it('should handle resource content with text', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const resourceContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [{
              type: 'resource',
              resource: {
                uri: 'file:///data/results.csv',
                text: 'name,value\ntemp,22\nhumidity,45',
              },
            }],
          },
        },
      }];

      mockCallTool.mockResolvedValue(resourceContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: '[Resource: file:///data/results.csv]\nname,value\ntemp,22\nhumidity,45' },
      ]);
    });

    it('should handle resource content with blob', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const resourceContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [{
              type: 'resource',
              resource: {
                uri: 'https://example.com/image.png',
                blob: 'base64ImageData',
                mimeType: 'image/png',
              },
            }],
          },
        },
      }];

      mockCallTool.mockResolvedValue(resourceContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        { inlineData: { data: 'base64ImageData', mimeType: 'image/png' } },
      ]);
    });

    it('should handle resource with only URI', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const resourceContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [{
              type: 'resource',
              resource: {
                uri: 'https://example.com/data.json',
              },
            }],
          },
        },
      }];

      mockCallTool.mockResolvedValue(resourceContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: '[Resource: https://example.com/data.json]' },
      ]);
    });

    it('should handle error responses', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const errorResponse: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            isError: true,
            content: [
              { type: 'text', text: 'File not found' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(errorResponse);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `Error from ${serverToolName}: File not found` },
      ]);
    });

    it('should handle invalid media content', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const invalidMedia: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'image', data: null as any, mimeType: 'image/png' }, // Missing data
              { type: 'image', data: 'someData', mimeType: null as any }, // Missing mimeType
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(invalidMedia);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: '[Invalid media: missing data or mimeType]' },
        { text: '[Invalid media: missing data or mimeType]' },
      ]);
    });

    it('should handle unknown content types', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const unknownContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'custom', data: 'someData' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(unknownContent);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: '[Unknown content type: custom]' },
      ]);
    });

    it('should handle multiple function response parts', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const multipleParts: Part[] = [
        {
          functionResponse: {
            name: serverToolName,
            response: {
              content: [{ type: 'text', text: 'First response' }],
            },
          },
        },
        {
          functionResponse: {
            name: serverToolName,
            response: {
              content: [{ type: 'image', data: 'imageData', mimeType: 'image/png' }],
            },
          },
        },
      ];

      mockCallTool.mockResolvedValue(multipleParts);
      const result: ToolResult = await tool.execute({});

      // The first part has no media, the second part has media and gets context prepended
      expect(result.llmContent).toEqual([
        { text: 'First response' },
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        { inlineData: { data: 'imageData', mimeType: 'image/png' } },
      ]);
    });

    it('should maintain backward compatibility for text-only responses', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const textOnlyResponse: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { text: 'Simple text response' }, // Old format without type
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(textOnlyResponse);
      const result: ToolResult = await tool.execute({});

      // Should keep original functionResponse for backward compatibility
      expect(result.llmContent).toEqual(textOnlyResponse);
    });

    it('should handle empty content array', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const emptyContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [],
          },
        },
      }];

      mockCallTool.mockResolvedValue(emptyContent);
      const result: ToolResult = await tool.execute({});

      // Should keep original functionResponse when no transformation needed
      expect(result.llmContent).toEqual(emptyContent);
    });

    it('should handle all media types (image, audio, video, pdf)', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const allMediaTypes: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'image', data: 'imageData', mimeType: 'image/png' },
              { type: 'audio', data: 'audioData', mimeType: 'audio/mp3' },
              { type: 'video', data: 'videoData', mimeType: 'video/mp4' },
              { type: 'pdf', data: 'pdfData', mimeType: 'application/pdf' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(allMediaTypes);
      const result: ToolResult = await tool.execute({});

      expect(result.llmContent).toEqual([
        { text: `[Response from ${serverToolName} (${serverName} MCP Server)]` },
        { inlineData: { data: 'imageData', mimeType: 'image/png' } },
        { inlineData: { data: 'audioData', mimeType: 'audio/mp3' } },
        { inlineData: { data: 'videoData', mimeType: 'video/mp4' } },
        { inlineData: { data: 'pdfData', mimeType: 'application/pdf' } },
      ]);
    });

    it('should handle display for media content', async () => {
      const tool = new DiscoveredMCPTool(
        mockCallableToolInstance,
        serverName,
        serverToolName,
        baseDescription,
        inputSchema,
      );

      const mediaContent: Part[] = [{
        functionResponse: {
          name: serverToolName,
          response: {
            content: [
              { type: 'text', text: 'Results:' },
              { type: 'image', data: 'imageData', mimeType: 'image/png' },
              { type: 'pdf', data: 'pdfData', mimeType: 'application/pdf' },
            ],
          },
        },
      }];

      mockCallTool.mockResolvedValue(mediaContent);
      const result: ToolResult = await tool.execute({});

      // Check display formatting
      expect(result.returnDisplay).toContain('Results:');
      expect(result.returnDisplay).toContain('[Image: image/png]');
      expect(result.returnDisplay).toContain('[PDF document]');
    });
  });
});
