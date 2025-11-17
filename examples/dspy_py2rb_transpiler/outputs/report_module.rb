# Feedback: Ruby outputs did not match Python results.
# Hints: Ruby outputs did not match Python results.
# DSPy summary: The module doubles integer scores, providing functions to scale individual scores and lists of scores, and formats the results as a string. There may be output discrepancies when compared to a Ruby version.
module Report
  module_function
  def scale_score(score)
    score * 2
  end
  def scale_scores(values)
    values.map { |value| scale_score(value) }
  end
  def render_report(values)
    scale_scores(values).map(&:to_s).join(", ")
  end
end