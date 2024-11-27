DATA_QUERY = """
SELECT * FROM (
    SELECT HADM_ID, STRING_AGG(CAST(TEXT AS STRING), ' ') AS text
    FROM `physionet-data.mimiciii_notes.noteevents`
    WHERE CATEGORY = "Discharge summary"
    GROUP BY HADM_ID
) as t1
JOIN (
  SELECT distinct HADM_ID
  FROM `physionet-data.mimiciii_clinical.diagnoses_icd` 
  where ICD9_CODE is not null
) as t2 on t1.HADM_ID = t2.HADM_ID
ORDER BY t1.HADM_ID
 """

LABELS_QUERY = """
select * from (
    SELECT HADM_ID, CAST(ICD9_CODE AS STRING) AS diagnosis
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` 
    where ICD9_CODE is not null
    # GROUP BY HADM_ID
    ) AS t1
JOIN (
SELECT distinct HADM_ID
FROM `physionet-data.mimiciii_notes.noteevents`
WHERE CATEGORY = "Discharge summary"
) as t2 on t1.HADM_ID = t2.HADM_ID
ORDER BY t1.HADM_ID
 """