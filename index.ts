import type { AdapterOptions } from "./types.js";
import type { CompletionAdapter } from "adminforth";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenAI } from "@google/genai";
import pRetry from 'p-retry';
import { logger } from "adminforth";

type CompletionRequestInput = {
  content: string;
  maxTokens?: number;
  outputSchema?: any;
};

type AgentModelPurpose = "primary" | "summary";

const GOOGLE_LANGCHAIN_AGENT_OPTION_KEYS = [
  "temperature",
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

export default class CompletionAdapterGoogleGemini
  implements CompletionAdapter
{
  options: AdapterOptions;

  constructor(options: AdapterOptions) {
    this.options = options;
  }
  
  validate() {
    if (!this.options.geminiApiKey) {
      throw new Error("geminiApiKey is required");
    }
  }

  async measureTokensCount(content: string): Promise<number> {
    // Implement token counting logic here
    const ai = new GoogleGenAI({
      apiKey: this.options.geminiApiKey,
    });
    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: content, 
    });

    return countTokensResponse.totalTokens;
  }

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    const modelOptions = getGoogleLangChainAgentOptions(
      this.options.extraRequestBodyParameters,
    );

    return {
      model: new ChatGoogleGenerativeAI({
        model: this.options.model || "gemini-3-flash-preview",
        apiKey: this.options.geminiApiKey,
        maxOutputTokens: params.maxTokens,
        ...modelOptions,
      } as any),
    };
  }


  complete = async (
    requestOrContent: CompletionRequestInput | string,
    stopOrMaxTokens: string[] | number = 50,
    maxTokensOrOutputSchema: number | any = 50,
    outputSchema?: any,
  ): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {
    const request =
      typeof requestOrContent === "string"
        ? {
            content: requestOrContent,
            maxTokens:
              typeof stopOrMaxTokens === "number"
                ? stopOrMaxTokens
                : typeof maxTokensOrOutputSchema === "number"
                  ? maxTokensOrOutputSchema
                  : 50,
            outputSchema:
              typeof stopOrMaxTokens === "number"
                ? maxTokensOrOutputSchema
                : outputSchema,
          }
        : requestOrContent;
    const {
      content,
      maxTokens: requestMaxTokens = 50,
      outputSchema: requestOutputSchema,
    } = request;

    const ai = new GoogleGenAI({
      apiKey: this.options.geminiApiKey,
    });

    const tryToGenerate = async () => {
      logger.debug("Making Google Gemini API call");
      try {
        const response = await ai.models.generateContent({
          model: this.options.model || "gemini-3-flash-preview",
          contents: content,
          config: {
            responseJsonSchema: requestOutputSchema ? requestOutputSchema : undefined,
            maxOutputTokens: requestMaxTokens,
            ...this.options.extraRequestBodyParameters,
          },
        });
        logger.debug(`Google Gemini SUCCESSFUL API response: ${response}`);
        return {
          content: response.text,
        };
      } catch (error) {
        logger.error(`Error during Google Gemini API call: ${error}`);
        throw new Error(`Error during Google Gemini API call: ${JSON.parse(error.message).error.message}`);
      }
    }
    const result = await pRetry(tryToGenerate, {
      retries: 5,
      onFailedAttempt: ({error, attemptNumber, retriesLeft, retriesConsumed}) => {
        logger.debug(`Attempt ${attemptNumber} failed. ${retriesLeft} retries left. ${retriesConsumed} retries consumed.`);
      },
    })
    return result;
  };
}
