module Report
  require 'json'

  def self.normalize_scores(values, min_size: 3)
    """Normalize a sequence by shifting values relative to the first element."""
    
    cleaned = values.compact.map(&:to_i)
    return [0] * min_size if cleaned.length < min_size

    base = cleaned.first
    cleaned.map { |value| value - base }
  end

  def self.generate_report(metrics, threshold: 3, fmt: "plain")
    """Return a textual (or JSON) report of winners over the threshold."""
    
    winners = metrics.select { |_, score| score >= threshold }.keys.sort
    if fmt == "json"
      return JSON.generate({ winners: winners, count: winners.length })
    end

    body = winners.empty? ? "n/a" : winners.join(", ")
    "Winners (#{winners.length}): #{body}"
  end
end