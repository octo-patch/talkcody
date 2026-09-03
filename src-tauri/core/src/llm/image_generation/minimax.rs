use crate::llm::auth::api_key_manager::{ApiKeyManager, ProviderCredentials};
use crate::llm::image_generation::types::{GeneratedImage, ImageGenerationRequest};
use crate::llm::types::ProviderConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const CHINA_IMAGE_ENDPOINT: &str = "https://api.minimaxi.com/v1/image_generation";
const INTERNATIONAL_IMAGE_ENDPOINT: &str = "https://api.minimax.io/v1/image_generation";
const MAX_PROMPT_CHARS: usize = 1_500;

#[derive(Debug, Clone, Serialize)]
struct MiniMaxImageRequest {
    model: String,
    prompt: String,
    aspect_ratio: String,
    response_format: String,
    n: u32,
    prompt_optimizer: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct MiniMaxImageResponse {
    data: Option<MiniMaxImageData>,
    base_resp: Option<MiniMaxBaseResponse>,
}

#[derive(Debug, Clone, Deserialize)]
struct MiniMaxImageData {
    #[serde(default)]
    image_urls: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct MiniMaxBaseResponse {
    status_code: i64,
    #[serde(default)]
    status_msg: String,
}

pub struct MiniMaxImageClient {
    config: ProviderConfig,
}

impl MiniMaxImageClient {
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
    }

    fn endpoint_for_region(use_international: bool) -> &'static str {
        if use_international {
            INTERNATIONAL_IMAGE_ENDPOINT
        } else {
            CHINA_IMAGE_ENDPOINT
        }
    }

    fn endpoint_from_custom_base_url(base_url: &str) -> String {
        let base_url = base_url.trim().trim_end_matches('/');
        if base_url.ends_with("/v1/image_generation") {
            return base_url.to_string();
        }

        for suffix in ["/anthropic/v1", "/anthropic", "/v1"] {
            if let Some(root) = base_url.strip_suffix(suffix) {
                return format!("{root}/v1/image_generation");
            }
        }

        format!("{base_url}/v1/image_generation")
    }

    async fn resolve_endpoint(&self, api_keys: &ApiKeyManager) -> Result<String, String> {
        let custom_url_key = format!("base_url_{}", self.config.id);
        if let Some(custom_url) = api_keys.get_setting(&custom_url_key).await? {
            if !custom_url.trim().is_empty() {
                return Ok(Self::endpoint_from_custom_base_url(&custom_url));
            }
        }

        let international_key = format!("use_international_{}", self.config.id);
        let use_international = matches!(
            api_keys.get_setting(&international_key).await?.as_deref(),
            Some("true")
        );
        Ok(Self::endpoint_for_region(use_international).to_string())
    }

    fn aspect_ratio(size: Option<&str>) -> &'static str {
        match size {
            Some("1792x1024" | "1280x720") => "16:9",
            Some("1152x864") => "4:3",
            Some("1248x832") => "3:2",
            Some("832x1248") => "2:3",
            Some("864x1152") => "3:4",
            Some("1024x1792" | "720x1280") => "9:16",
            Some("1344x576") => "21:9",
            _ => "1:1",
        }
    }

    fn build_request(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<MiniMaxImageRequest, String> {
        let prompt_chars = request.prompt.chars().count();
        if prompt_chars == 0 || prompt_chars > MAX_PROMPT_CHARS {
            return Err(format!(
                "MiniMax image prompts must contain between 1 and {MAX_PROMPT_CHARS} characters"
            ));
        }

        let n = request.n.unwrap_or(1);
        if !(1..=9).contains(&n) {
            return Err("MiniMax image count must be between 1 and 9".to_string());
        }

        Ok(MiniMaxImageRequest {
            model: model.to_string(),
            prompt: request.prompt,
            aspect_ratio: Self::aspect_ratio(request.size.as_deref()).to_string(),
            response_format: "url".to_string(),
            n,
            prompt_optimizer: true,
        })
    }

    fn into_images(payload: MiniMaxImageResponse) -> Result<Vec<GeneratedImage>, String> {
        if let Some(base_resp) = payload.base_resp {
            if base_resp.status_code != 0 {
                return Err(format!(
                    "MiniMax image generation failed ({}): {}",
                    base_resp.status_code, base_resp.status_msg
                ));
            }
        }

        let image_urls = payload.data.map(|data| data.image_urls).unwrap_or_default();
        if image_urls.is_empty() {
            return Err("MiniMax image generation returned no images".to_string());
        }

        Ok(image_urls
            .into_iter()
            .map(|url| GeneratedImage {
                b64_json: None,
                url: Some(url),
                mime_type: "image/png".to_string(),
                revised_prompt: None,
            })
            .collect())
    }

    pub async fn generate(
        &self,
        api_keys: &ApiKeyManager,
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<Vec<GeneratedImage>, String> {
        let api_key = match api_keys.get_credentials(&self.config).await? {
            ProviderCredentials::Token(token) => token,
            ProviderCredentials::None => {
                return Err("API key not configured for MiniMax image generation".to_string())
            }
        };

        let endpoint = self.resolve_endpoint(api_keys).await?;
        let body = Self::build_request(model, request)?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| format!("Failed to build HTTP client: {error}"))?;

        let response = client
            .post(endpoint)
            .bearer_auth(api_key)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("MiniMax image request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!(
                "MiniMax image generation failed ({status}): {body}"
            ));
        }

        let payload = response
            .json::<MiniMaxImageResponse>()
            .await
            .map_err(|error| format!("Failed to parse MiniMax image response: {error}"))?;
        Self::into_images(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(size: Option<&str>) -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "image-01".to_string(),
            prompt: "A lighthouse above a quiet sea".to_string(),
            size: size.map(str::to_string),
            quality: Some("standard".to_string()),
            n: Some(2),
            response_format: Some("url".to_string()),
            provider_options: None,
            request_id: None,
        }
    }

    #[test]
    fn selects_regional_endpoints() {
        assert_eq!(
            MiniMaxImageClient::endpoint_for_region(false),
            CHINA_IMAGE_ENDPOINT
        );
        assert_eq!(
            MiniMaxImageClient::endpoint_for_region(true),
            INTERNATIONAL_IMAGE_ENDPOINT
        );
    }

    #[test]
    fn normalizes_custom_provider_endpoints() {
        assert_eq!(
            MiniMaxImageClient::endpoint_from_custom_base_url("https://example.com/anthropic/v1/"),
            "https://example.com/v1/image_generation"
        );
        assert_eq!(
            MiniMaxImageClient::endpoint_from_custom_base_url(
                "https://example.com/v1/image_generation"
            ),
            "https://example.com/v1/image_generation"
        );
    }

    #[test]
    fn builds_minimax_request_schema() {
        let body = MiniMaxImageClient::build_request("image-01", request(Some("1792x1024")))
            .expect("build request");
        assert_eq!(
            serde_json::to_value(body).expect("serialize request"),
            json!({
                "model": "image-01",
                "prompt": "A lighthouse above a quiet sea",
                "aspect_ratio": "16:9",
                "response_format": "url",
                "n": 2,
                "prompt_optimizer": true
            })
        );
    }

    #[test]
    fn parses_image_urls() {
        let payload: MiniMaxImageResponse = serde_json::from_value(json!({
            "data": {"image_urls": ["https://example.com/one.png", "https://example.com/two.png"]},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("parse response");

        let images = MiniMaxImageClient::into_images(payload).expect("convert response");
        assert_eq!(images.len(), 2);
        assert_eq!(
            images[0].url.as_deref(),
            Some("https://example.com/one.png")
        );
    }

    #[test]
    fn surfaces_api_errors() {
        let payload: MiniMaxImageResponse = serde_json::from_value(json!({
            "base_resp": {"status_code": 1008, "status_msg": "invalid request"}
        }))
        .expect("parse response");

        let error = MiniMaxImageClient::into_images(payload).expect_err("reject response");
        assert!(error.contains("1008"));
        assert!(error.contains("invalid request"));
    }
}
