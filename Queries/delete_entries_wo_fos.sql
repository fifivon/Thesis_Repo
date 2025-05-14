DELETE FROM embeddings
WHERE paper_id IN (
    SELECT id FROM papers
    WHERE array_length(fos_names, 1) IS NULL OR array_length(fos_names, 1) = 0
);

DELETE FROM papers
WHERE array_length(fos_names, 1) IS NULL OR array_length(fos_names, 1) = 0;