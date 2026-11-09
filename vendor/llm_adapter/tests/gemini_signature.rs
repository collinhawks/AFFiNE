use llm_adapter::{
    backend::BackendRequestLayer,
    core::{CoreContent, CoreMessage, CoreRequest, CoreRole, StreamEvent},
    protocol::gemini,
    stream::parse_gemini_stream,
};
use serde_json::json;

#[test]
fn gemini_signature_survives_stream_and_tool_replay() {
    // Simulate two parallel Gemini function calls.
    // Only the first has a thought signature.
    let response = json!({
        "responseId": "test_response",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "doc_search",
                            "args": {"query": "Untitled-2"}
                        },
                        "thoughtSignature": "test_signature_123"
                    },
                    {
                        "functionCall": {
                            "name": "artifact_search",
                            "args": {"query": "notes"}
                        }
                    }
                ]
            },
            "finishReason": "STOP"
        }]
    });

    let raw = format!("data: {response}\n\ndata: [DONE]\n\n");

    let calls: Vec<_> = parse_gemini_stream(&raw)
        .unwrap()
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCall {
                call_id,
                name,
                arguments,
                thought,
            } => Some((call_id, name, arguments, thought)),
            _ => None,
        })
        .collect();

    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].3.as_deref(), Some("test_signature_123"));
    assert_eq!(calls[1].3, None);

    // Verify non-streaming decoding preserves the same signature.
    let decoded = gemini::response::decode(&response).unwrap();

    assert!(decoded.message.content.iter().any(|part| {
        matches!(
            part,
            CoreContent::ToolCall {
                thought: Some(signature),
                ..
            } if signature == "test_signature_123"
        )
    }));

    // Reconstruct the assistant tool calls and tool results.
    let request = CoreRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: vec![
            CoreMessage {
                role: CoreRole::Assistant,
                content: calls
                    .iter()
                    .map(|(id, name, args, signature)| {
                        CoreContent::ToolCall {
                            call_id: id.clone(),
                            name: name.clone(),
                            arguments: args.clone(),
                            thought: signature.clone(),
                        }
                    })
                    .collect(),
            },
            CoreMessage {
                role: CoreRole::Tool,
                content: calls
                    .iter()
                    .map(|(id, _, _, _)| CoreContent::ToolResult {
                        call_id: id.clone(),
                        output: json!({"ok": true}),
                        is_error: Some(false),
                    })
                    .collect(),
            },
        ],
        stream: false,
        max_tokens: None,
        temperature: None,
        tools: vec![],
        tool_choice: None,
        include: None,
        reasoning: None,
        response_schema: None,
    };

    let encoded = gemini::request::encode(
        &request,
        false,
        BackendRequestLayer::GeminiApi,
        "https://generativelanguage.googleapis.com/v1beta",
    );

    // The original signature must remain attached to doc_search.
    assert_eq!(
        encoded["contents"][0]["parts"][0]["thoughtSignature"],
        "test_signature_123"
    );

    // No signature should be invented for artifact_search.
    assert!(
        encoded["contents"][0]["parts"][1]
            .get("thoughtSignature")
            .is_none()
    );

    // The tool results must retain their correct names.
    assert_eq!(
        encoded["contents"][1]["parts"][0]["functionResponse"]["name"],
        "doc_search"
    );
    assert_eq!(
        encoded["contents"][1]["parts"][1]["functionResponse"]["name"],
        "artifact_search"
    );
}
