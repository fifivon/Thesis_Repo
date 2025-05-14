SELECT COUNT(DISTINCT venue) AS distinct_venues
FROM papers
WHERE venue IS NOT NULL;
