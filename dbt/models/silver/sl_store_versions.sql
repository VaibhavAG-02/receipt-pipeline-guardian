-- Silver: close out each store attribute version.
--
-- This is the load-bearing step for the Type 2 dimension. `lead()` supplies the
-- next version's effective date, which becomes this version's expiry. The open
-- interval [valid_from, valid_to) is what makes a point-in-time join correct --
-- a half-open interval avoids the double-count you get when a receipt lands
-- exactly on a change boundary.
with src as (
    select
        store_id,
        cast(effective_from as timestamp) as effective_from,
        region,
        store_format,
        manager_id,
        change_reason
    from {{ ref('br_store_master') }}
),
versioned as (
    select
        *,
        lead(effective_from) over (
            partition by store_id order by effective_from
        ) as next_effective_from,
        row_number() over (
            partition by store_id order by effective_from
        ) as version_number
    from src
)
select
    store_id,
    version_number,
    -- The first version is opened to the beginning of time rather than to its
    -- own effective date. Without this, any receipt older than the initial
    -- load -- device clock skew produces plenty -- matches no version at all
    -- and silently drops out of the fact table on the point-in-time join.
    -- Caught by assert_point_in_time_join_is_correct, which is why it is there.
    case when version_number = 1
         then timestamp '1900-01-01'
         else effective_from
    end                                                     as valid_from,
    coalesce(next_effective_from, timestamp '9999-12-31')    as valid_to,
    next_effective_from is null                              as is_current,
    region,
    store_format,
    manager_id,
    change_reason
from versioned
