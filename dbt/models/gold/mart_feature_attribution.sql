-- Global SHAP ranking. Reported next to the model's gain-based importance
-- because the two routinely disagree, and the disagreement is informative.
select feature, mean_abs_shap
from {{ ref('br_shap_global') }}
order by mean_abs_shap desc
