SELECT COUNT(DISTINCT normalized_venue) AS distinct_normalized_venues
FROM venues_norm
WHERE normalized_venue IS NOT NULL AND normalized_venue <> '';