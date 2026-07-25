-- Type 2 store dimension.
--
-- One row per (store, version). `store_key` is a surrogate hash of the natural
-- key plus the version's start, so a fact row pins to the store *as it was*
-- when the receipt was submitted. Joining on store_id alone would restate
-- history every time a store is remodelled or a district is redrawn -- last
-- quarter's numbers would silently change.
select
    md5(store_id || '|' || cast(valid_from as varchar)) as store_key,
    store_id                                            as store_natural_key,
    version_number,
    valid_from,
    valid_to,
    is_current,
    region,
    store_format,
    manager_id,
    change_reason
from {{ ref('sl_store_versions') }}
