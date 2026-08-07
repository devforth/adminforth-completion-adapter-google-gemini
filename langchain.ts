import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import type { AdapterOptions } from "./types.js";

export type AgentModelPurpose = "primary" | "summary";

const GOOGLE_LANGCHAIN_AGENT_OPTION_KEYS = [
  "temperature",
  "maxOutputTokens",
  "topP",
  "topK",
  "stopSequences",
  "safetySettings",
  "apiVersion",
  "baseUrl",
  "customHeaders",
  "streaming",
  "json",
  "streamUsage",
  "convertSystemMessageToHumanContent",
  "thinkingConfig",
] as const;

function getGoogleLangChainAgentOptions(
  extraRequestBodyParameters?: Record<string, unknown>,
) {
  const options: Record<string, unknown> = {};

  if (!extraRequestBodyParameters) {
    return options;
  }

  for (const key of GOOGLE_LANGCHAIN_AGENT_OPTION_KEYS) {
    if (key in extraRequestBodyParameters) {
      options[key] = extraRequestBodyParameters[key];
    }
  }

  return options;
}

const GEMINI_SCHEMA_KEYS = new Set([
  "description",
  "enum",
  "format",
  "items",
  "maxItems",
  "minItems",
  "nullable",
  "properties",
  "required",
  "type",
]);

function sanitizeGeminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(sanitizeGeminiSchema);
  }

  if (!schema || typeof schema !== "object") {
    return schema;
  }

  const sanitizedSchema: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (!GEMINI_SCHEMA_KEYS.has(key)) {
      continue;
    }

    if (key === "properties" && value && typeof value === "object") {
      sanitizedSchema.properties = Object.fromEntries(
        Object.entries(value).map(([propertyName, propertySchema]) => [
          propertyName,
          sanitizeGeminiSchema(propertySchema),
        ]),
      );
      continue;
    }

    sanitizedSchema[key] = sanitizeGeminiSchema(value);
  }

  return sanitizedSchema;
}

function sanitizeGeminiTools(tools: unknown): unknown {
  if (!Array.isArray(tools)) {
    return tools;
  }

  return tools.map((tool) => {
    if (!tool || typeof tool !== "object" || !Array.isArray(tool.functionDeclarations)) {
      return tool;
    }

    return {
      ...tool,
      functionDeclarations: tool.functionDeclarations.map((declaration: any) => ({
        ...declaration,
        ...(declaration.parameters
          ? { parameters: sanitizeGeminiSchema(declaration.parameters) }
          : {}),
      })),
    };
  });
}

export function createLangChainAgentSpec(params: {
  options: AdapterOptions;
  maxTokens: number;
  purpose: AgentModelPurpose;
}) {
  const modelOptions = getGoogleLangChainAgentOptions(
    params.options.extraRequestBodyParameters,
  );

  const model = new ChatGoogleGenerativeAI({
      model: params.options.model || "gemini-3-flash-preview",
      apiKey: params.options.geminiApiKey,
      maxOutputTokens: params.maxTokens,
      ...modelOptions,
    } as any);
  const getInvocationParams = model.invocationParams.bind(model);
  model.invocationParams = ((options) => {
    const request = getInvocationParams(options);

    if (!request.tools) {
      return request;
    }

    return {
      ...request,
      tools: sanitizeGeminiTools(request.tools) as typeof request.tools,
    };
  }) as typeof model.invocationParams;

  return {
    model,
  };
}
