import type { CompletionAdapter } from "adminforth";
import type { AdapterOptions } from "./types.js";
import {
  GeminiService,
  type CompletionRequestInput,
  type CompletionResult,
} from "./gemini.js";
import {
  createLangChainAgentSpec,
  type AgentModelPurpose,
} from "./langchain.js";

export type { AdapterOptions } from "./types.js";

class CompletionAdapterGoogleGemini implements CompletionAdapter {
  options: AdapterOptions;
  private gemini: GeminiService;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.gemini = new GeminiService(options);
  }

  validate() {
    this.gemini.validate();
  }

  measureTokensCount(content: string): Promise<number> {
    return this.gemini.measureTokensCount(content);
  }

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    return createLangChainAgentSpec({
      options: this.options,
      maxTokens: params.maxTokens,
      purpose: params.purpose,
    });
  }

  complete = async (
    requestOrContent: CompletionRequestInput | string,
    stopOrMaxTokens: string[] | number = 50,
    maxTokensOrOutputSchema: number | any = 50,
    outputSchema?: any,
  ): Promise<CompletionResult> => {
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

    return this.gemini.complete(request);
  };
}

export { CompletionAdapterGoogleGemini };
export default CompletionAdapterGoogleGemini;
