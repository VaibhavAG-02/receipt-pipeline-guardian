-- Bronze: typed passthrough of the landing zone. No business logic here on
-- purpose -- if a rule lives in bronze you can never replay raw history.
select * from raw.scores
