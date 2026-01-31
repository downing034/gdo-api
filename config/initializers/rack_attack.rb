# frozen_string_literal: true

class Rack::Attack
  throttle("req/ip", limit: 300, period: 5.minutes) do |req|
    req.ip
  end

  throttle("simulate/ip", limit: 20, period: 1.minute) do |req|
    req.ip if req.path == "/api/v1/predictions/simulate" && req.post?
  end

  self.throttled_responder = lambda do |req|
    retry_after = req.env["rack.attack.match_data"][:period]
    [
      429,
      { "Content-Type" => "application/json" },
      [{
        error: {
          code: "rate_limit_exceeded",
          message: "Too many requests",
          details: { retry_after: retry_after }
        }
      }.to_json]
    ]
  end
end