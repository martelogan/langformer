"""Stub translation used for deterministic tests."""

STUB_RUBY_IMPLEMENTATION = """\
module Report
  def self.normalize_scores(values, min_size: 3)
    cleaned = values.compact.map { |v| v.to_i }
    return Array.new(min_size, 0) if cleaned.length < min_size
    base = cleaned.first
    cleaned.map { |v| v - base }
  end

  def self.generate_report(metrics, threshold: 3, fmt: "plain")
    normalized = {}
    metrics.each { |k, v| normalized[k.to_s] = v }
    winners = normalized.keys.sort.select { |key| normalized[key].to_i >= threshold }

    if fmt == "json"
      require "json"
      return JSON.generate({ "winners" => winners, "count" => winners.length })
    end

    body = winners.empty? ? "n/a" : winners.join(", ")
    "Winners (#{winners.length}): #{body}"
  end
end
"""
