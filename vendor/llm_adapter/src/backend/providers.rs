use super::{BackendOperation, BackendProvider, BackendRequestLayer, OpenAiDialect, TargetCompileError};

impl BackendProvider {
  pub fn matches_registry_backend(self, backend: &str) -> bool {
    match self {
      Self::OpenAi => matches!(backend, "openai_responses" | "openai_chat"),
      Self::Anthropic => backend == "anthropic",
      Self::AnthropicVertex => backend == "anthropic_vertex",
      Self::Gemini => backend == "gemini_api",
      Self::GeminiVertex => backend == "gemini_vertex",
      Self::Fal => backend == "fal",
      Self::CloudflareWorkersAi => backend == "cloudflare_workers_ai",
      Self::Openrouter => backend == "openrouter",
      Self::Perplexity => backend == "perplexity",
    }
  }

  pub fn default_request_layer(
    self,
    operation: BackendOperation,
    dialect: Option<OpenAiDialect>,
  ) -> Result<BackendRequestLayer, TargetCompileError> {
    use BackendOperation as Op;
    use BackendRequestLayer as Layer;
    if dialect.is_some() && self != Self::OpenAi {
      return Err(TargetCompileError::InvalidConfig(
        "OpenAI dialect is only valid for the OpenAI provider".into(),
      ));
    }
    let layer = match self {
      Self::OpenAi => match operation {
        Op::Embedding => Layer::OpenaiEmbeddings,
        Op::Rerank => Layer::ChatCompletions,
        Op::Image => Layer::OpenaiImages,
        Op::Chat | Op::Structured => match dialect.unwrap_or(OpenAiDialect::Responses) {
          OpenAiDialect::Responses => Layer::Responses,
          OpenAiDialect::ChatCompletions => Layer::ChatCompletions,
        },
      },
      Self::Anthropic => Layer::Anthropic,
      Self::AnthropicVertex => Layer::VertexAnthropic,
      Self::Gemini => Layer::GeminiApi,
      Self::GeminiVertex => Layer::GeminiVertex,
      Self::Fal => Layer::Fal,
      Self::CloudflareWorkersAi => Layer::CloudflareWorkersAi,
      Self::Openrouter => {
        if operation == Op::Rerank {
          Layer::OpenrouterChatCompletions
        } else {
          Layer::OpenrouterResponses
        }
      }
      Self::Perplexity => Layer::PerplexitySonar,
    };
    Ok(layer)
  }
}

impl BackendProvider {
  pub fn configured_layer_matches(
    self,
    config: &serde_json::Value,
    layer: BackendRequestLayer,
    operation: BackendOperation,
  ) -> Result<bool, TargetCompileError> {
    if self == Self::OpenAi
      && matches!(operation, BackendOperation::Chat | BackendOperation::Structured)
      && let Some(value) = config.get("openaiDialect").filter(|value| !value.is_null())
    {
      let dialect =
        serde_json::from_value(value.clone()).map_err(|error| TargetCompileError::InvalidConfig(error.to_string()))?;
      return Ok(layer == self.default_request_layer(operation, Some(dialect))?);
    }
    Ok(true)
  }
}
