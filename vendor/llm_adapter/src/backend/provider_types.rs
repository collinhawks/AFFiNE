use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::TargetCompileError;

#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendProvider {
  #[serde(rename = "openai")]
  OpenAi,
  #[serde(rename = "anthropic")]
  Anthropic,
  #[serde(rename = "anthropicVertex")]
  AnthropicVertex,
  #[serde(rename = "gemini")]
  Gemini,
  #[serde(rename = "geminiVertex")]
  GeminiVertex,
  #[serde(rename = "fal")]
  Fal,
  #[serde(rename = "cloudflareWorkersAi")]
  CloudflareWorkersAi,
  #[serde(rename = "openrouter")]
  Openrouter,
  #[serde(rename = "perplexity")]
  Perplexity,
}

impl BackendProvider {
  pub const ALL: [Self; 9] = [
    Self::OpenAi,
    Self::Anthropic,
    Self::AnthropicVertex,
    Self::Gemini,
    Self::GeminiVertex,
    Self::Fal,
    Self::CloudflareWorkersAi,
    Self::Openrouter,
    Self::Perplexity,
  ];

  pub fn as_str(self) -> &'static str {
    self.definition().id
  }
}

impl FromStr for BackendProvider {
  type Err = TargetCompileError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    Self::ALL
      .into_iter()
      .find(|provider| provider.as_str() == value)
      .ok_or_else(|| TargetCompileError::InvalidConfig(format!("unsupported backend provider: {value}")))
  }
}

#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiShape {
  Responses,
  ChatCompletions,
  Anthropic,
  VertexAnthropic,
  GeminiApi,
  GeminiVertex,
  OpenaiEmbeddings,
  OpenaiImages,
  CloudflareWorkersAi,
  Fal,
}

impl FromStr for ApiShape {
  type Err = TargetCompileError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    serde_json::from_value(Value::String(value.to_string()))
      .map_err(|_| TargetCompileError::InvalidConfig(format!("unsupported API shape: {value}")))
  }
}

#[derive(Clone, PartialEq, Eq)]
pub enum CredentialPlan {
  Field {
    field: String,
  },
  GoogleOAuth {
    scopes: Vec<String>,
    project: String,
    credentials: Option<Value>,
  },
}

impl fmt::Debug for CredentialPlan {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Field { field } => formatter.debug_struct("Field").field("field", field).finish(),
      Self::GoogleOAuth {
        scopes,
        project,
        credentials,
      } => formatter
        .debug_struct("GoogleOAuth")
        .field("scopes", scopes)
        .field("project", project)
        .field("credentials", &credentials.as_ref().map(|_| "[REDACTED]"))
        .finish(),
    }
  }
}
