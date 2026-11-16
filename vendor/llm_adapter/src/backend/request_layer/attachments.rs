use super::{AttachmentReferenceMode, AttachmentReferencePlan, AttachmentReferenceReason};

pub(super) fn generic_remote_reference_plan(url: &str) -> AttachmentReferencePlan {
  let Ok(parsed) = url.parse::<url::Url>() else {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  };

  if !matches!(parsed.scheme(), "http" | "https" | "gs") {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  }

  AttachmentReferencePlan {
    mode: AttachmentReferenceMode::Remote,
    reason: AttachmentReferenceReason::GenericRemoteReference,
  }
}

pub(super) fn plan_gemini_api_attachment_reference(url: &str, base_url: &str) -> AttachmentReferencePlan {
  let Ok(parsed) = url.parse::<url::Url>() else {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  };

  if !matches!(parsed.scheme(), "http" | "https") {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  }

  if is_gemini_file_url(&parsed, base_url) {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Remote,
      reason: AttachmentReferenceReason::GeminiApiFileUri,
    };
  }

  if is_youtube_url(&parsed) {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Remote,
      reason: AttachmentReferenceReason::GeminiApiYoutubeUrl,
    };
  }

  AttachmentReferencePlan {
    mode: AttachmentReferenceMode::Inline,
    reason: AttachmentReferenceReason::GeminiApiInlineHttpUrl,
  }
}

fn is_youtube_url(url: &url::Url) -> bool {
  let hostname = url.host_str().unwrap_or_default().to_ascii_lowercase();
  if hostname == "youtu.be" {
    let path = url.path().trim_matches('/');
    return !path.is_empty()
      && path
        .chars()
        .all(|char| char.is_ascii_alphanumeric() || char == '-' || char == '_');
  }

  if hostname != "youtube.com" && hostname != "www.youtube.com" {
    return false;
  }

  url.path() == "/watch" && url.query_pairs().any(|(key, value)| key == "v" && !value.is_empty())
}

fn is_gemini_file_url(url: &url::Url, base_url: &str) -> bool {
  let Ok(base) = base_url.parse::<url::Url>() else {
    return false;
  };

  if url.origin() != base.origin() {
    return false;
  }

  let base_path = base.path().trim_end_matches('/');
  let expected_prefix = format!("{base_path}/files/");
  url.path().starts_with(&expected_prefix)
}
