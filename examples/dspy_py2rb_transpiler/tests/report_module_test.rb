require "minitest/autorun"

candidate_path = ENV.fetch("CANDIDATE_PATH")
load candidate_path

class ReportTest < Minitest::Test
  def test_scale_score
    assert_equal 4, Report.scale_score(2)
    assert_equal 0, Report.scale_score(0)
  end

  def test_scale_scores
    assert_equal [2, 4, 6], Report.scale_scores([1, 2, 3])
  end

  def test_render_report
    output = Report.render_report([1, 2, 3])
    assert_match(/2, 4, 6/, output)
  end
end
