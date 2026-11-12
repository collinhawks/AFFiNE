use super::{
  ApiShape, BackendError, BackendOperation, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, ExecutableProtocol,
  ImageProtocol, RerankProtocol, StructuredProtocol,
};

impl ChatProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      ChatProtocol::OpenaiChatCompletions => "openai_chat_completions",
      ChatProtocol::OpenaiResponses => "openai_responses",
      ChatProtocol::AnthropicMessages => "anthropic_messages",
      ChatProtocol::GeminiGenerateContent => "gemini_generate_content",
    }
  }
}

impl StructuredProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      StructuredProtocol::OpenaiChatCompletions => "openai_chat_completions",
      StructuredProtocol::OpenaiResponses => "openai_responses",
      StructuredProtocol::GeminiGenerateContent => "gemini_generate_content",
    }
  }
}

impl EmbeddingProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      EmbeddingProtocol::Openai => "openai",
      EmbeddingProtocol::Gemini => "gemini",
    }
  }
}

impl RerankProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      RerankProtocol::OpenaiChatLogprobs => "openai_chat_logprobs",
      RerankProtocol::CloudflareWorkersAi => "cloudflare_workers_ai",
    }
  }
}

impl ImageProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      ImageProtocol::OpenaiImages => "openai_images",
      ImageProtocol::GeminiGenerateContent => "gemini_generate_content",
      ImageProtocol::FalImage => "fal_image",
    }
  }
}

impl BackendRequestLayer {
  pub fn executable_protocol(self, operation: BackendOperation) -> Result<ExecutableProtocol, BackendError> {
    use BackendOperation as Op;
    use ExecutableProtocol as Protocol;
    let protocol = match (self, operation) {
      (Self::Responses | Self::OpenrouterResponses, Op::Chat) => Protocol::Chat(ChatProtocol::OpenaiResponses),
      (
        Self::ChatCompletions | Self::OpenrouterChatCompletions | Self::CloudflareWorkersAi | Self::PerplexitySonar,
        Op::Chat,
      ) => Protocol::Chat(ChatProtocol::OpenaiChatCompletions),
      (Self::Anthropic | Self::VertexAnthropic | Self::OpenrouterAnthropic, Op::Chat) => {
        Protocol::Chat(ChatProtocol::AnthropicMessages)
      }
      (Self::GeminiApi | Self::GeminiVertex, Op::Chat) => Protocol::Chat(ChatProtocol::GeminiGenerateContent),
      (Self::Responses | Self::OpenrouterResponses, Op::Structured) => {
        Protocol::Structured(StructuredProtocol::OpenaiResponses)
      }
      (Self::ChatCompletions | Self::OpenrouterChatCompletions | Self::CloudflareWorkersAi, Op::Structured) => {
        Protocol::Structured(StructuredProtocol::OpenaiChatCompletions)
      }
      (Self::GeminiApi | Self::GeminiVertex, Op::Structured) => {
        Protocol::Structured(StructuredProtocol::GeminiGenerateContent)
      }
      (Self::Responses | Self::ChatCompletions | Self::CloudflareWorkersAi | Self::OpenaiEmbeddings, Op::Embedding) => {
        Protocol::Embedding(EmbeddingProtocol::Openai)
      }
      (Self::GeminiApi | Self::GeminiVertex, Op::Embedding) => Protocol::Embedding(EmbeddingProtocol::Gemini),
      (Self::ChatCompletions | Self::OpenrouterChatCompletions, Op::Rerank) => {
        Protocol::Rerank(RerankProtocol::OpenaiChatLogprobs)
      }
      (Self::CloudflareWorkersAi, Op::Rerank) => Protocol::Rerank(RerankProtocol::CloudflareWorkersAi),
      (Self::OpenaiImages, Op::Image) => Protocol::Image(ImageProtocol::OpenaiImages),
      (Self::GeminiApi | Self::GeminiVertex, Op::Image) => Protocol::Image(ImageProtocol::GeminiGenerateContent),
      (Self::Fal, Op::Image) => Protocol::Image(ImageProtocol::FalImage),
      _ => {
        return Err(BackendError::InvalidConfig {
          message: format!("request_layer `{}` does not support {operation:?}", self.as_str()),
        });
      }
    };
    Ok(protocol)
  }

  pub fn api_shape(self) -> ApiShape {
    match self {
      Self::Responses | Self::OpenrouterResponses => ApiShape::Responses,
      Self::ChatCompletions | Self::OpenrouterChatCompletions | Self::PerplexitySonar => ApiShape::ChatCompletions,
      Self::Anthropic | Self::OpenrouterAnthropic => ApiShape::Anthropic,
      Self::VertexAnthropic => ApiShape::VertexAnthropic,
      Self::GeminiApi => ApiShape::GeminiApi,
      Self::GeminiVertex => ApiShape::GeminiVertex,
      Self::CloudflareWorkersAi => ApiShape::CloudflareWorkersAi,
      Self::OpenaiEmbeddings => ApiShape::OpenaiEmbeddings,
      Self::OpenaiImages => ApiShape::OpenaiImages,
      Self::Fal => ApiShape::Fal,
    }
  }
  pub(super) fn from_chat_protocol(protocol: &ChatProtocol) -> Self {
    match protocol {
      ChatProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      ChatProtocol::OpenaiResponses => Self::Responses,
      ChatProtocol::AnthropicMessages => Self::Anthropic,
      ChatProtocol::GeminiGenerateContent => Self::GeminiApi,
    }
  }

  pub(super) fn from_structured_protocol(protocol: &StructuredProtocol) -> Self {
    match protocol {
      StructuredProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      StructuredProtocol::OpenaiResponses => Self::Responses,
      StructuredProtocol::GeminiGenerateContent => Self::GeminiApi,
    }
  }

  pub(super) fn from_embedding_protocol(protocol: &EmbeddingProtocol) -> Self {
    match protocol {
      EmbeddingProtocol::Openai => Self::ChatCompletions,
      EmbeddingProtocol::Gemini => Self::GeminiApi,
    }
  }

  pub(super) fn from_rerank_protocol(protocol: &RerankProtocol) -> Self {
    match protocol {
      RerankProtocol::OpenaiChatLogprobs => Self::ChatCompletions,
      RerankProtocol::CloudflareWorkersAi => Self::CloudflareWorkersAi,
    }
  }

  pub(super) fn from_image_protocol(protocol: &ImageProtocol) -> Self {
    match protocol {
      ImageProtocol::OpenaiImages => Self::OpenaiImages,
      ImageProtocol::GeminiGenerateContent => Self::GeminiApi,
      ImageProtocol::FalImage => Self::Fal,
    }
  }

  pub(super) fn ensure_chat_compatible(&self, protocol: &ChatProtocol) -> Result<(), BackendError> {
    if self.executable_protocol(BackendOperation::Chat)? == ExecutableProtocol::Chat(*protocol) {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with chat protocol `{}`",
          self.as_str(),
          protocol.as_str()
        ),
      })
    }
  }
  pub(super) fn ensure_structured_compatible(&self, protocol: &StructuredProtocol) -> Result<(), BackendError> {
    if self.executable_protocol(BackendOperation::Structured)? == ExecutableProtocol::Structured(*protocol) {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with structured protocol `{}`",
          self.as_str(),
          protocol.as_str()
        ),
      })
    }
  }
  pub(super) fn ensure_embedding_compatible(&self, protocol: &EmbeddingProtocol) -> Result<(), BackendError> {
    if self.executable_protocol(BackendOperation::Embedding)? == ExecutableProtocol::Embedding(*protocol) {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with embedding protocol `{}`",
          self.as_str(),
          protocol.as_str()
        ),
      })
    }
  }
  pub(super) fn ensure_rerank_compatible(&self, protocol: &RerankProtocol) -> Result<(), BackendError> {
    if self.executable_protocol(BackendOperation::Rerank)? == ExecutableProtocol::Rerank(*protocol) {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with rerank protocol `{}`",
          self.as_str(),
          protocol.as_str()
        ),
      })
    }
  }
  pub(super) fn ensure_image_compatible(&self, protocol: &ImageProtocol) -> Result<(), BackendError> {
    if self.executable_protocol(BackendOperation::Image)? == ExecutableProtocol::Image(*protocol) {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with image protocol `{}`",
          self.as_str(),
          protocol.as_str()
        ),
      })
    }
  }
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      BackendRequestLayer::Anthropic => "anthropic",
      BackendRequestLayer::ChatCompletions => "chat_completions",
      BackendRequestLayer::CloudflareWorkersAi => "cloudflare_workers_ai",
      BackendRequestLayer::GeminiApi => "gemini_api",
      BackendRequestLayer::GeminiVertex => "gemini_vertex",
      BackendRequestLayer::PerplexitySonar => "perplexity_sonar",
      BackendRequestLayer::Responses => "responses",
      BackendRequestLayer::VertexAnthropic => "vertex_anthropic",
      BackendRequestLayer::OpenaiImages => "openai_images",
      BackendRequestLayer::Fal => "fal",
      BackendRequestLayer::OpenaiEmbeddings => "openai_embeddings",
      BackendRequestLayer::OpenrouterAnthropic => "openrouter_anthropic",
      BackendRequestLayer::OpenrouterResponses => "openrouter_responses",
      BackendRequestLayer::OpenrouterChatCompletions => "openrouter_chat_completions",
    }
  }
}
