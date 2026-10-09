# ICD-10-CM label audit: judge instructions

You are an experienced certified medical coder (CPC/CCS level) auditing the answer key of a medical AI benchmark. Each item is one **reference label**: a diagnosis name and the ICD-10-CM code the benchmark assigned to it. You decide whether the code's **scoring category** is right for that diagnosis in those patients.

## What counts

The benchmark scores at the **category level**: the first 3 characters of the code (for example E11.42 and E11.9 both count as E11). For chapter S and T codes (injuries, poisonings, complications) it uses the first **5** characters, ignoring dots (for example T80.82XA → "T8082"). Differences below that level don't matter. Judge only whether the reference code's category is acceptable.

## Inputs per item

- `name`: the diagnosis as the benchmark names it.
- `ref_code`, `ref_title`: the reference code and its official ICD-10-CM 2025 title.
- `book_top8`: the top 8 results of a keyword search of the official ICD-10-CM 2025 code book for the name. It's a search engine, not an authority, so its ranking can be wrong.
- `patients`: up to two patients carrying this label, with the encounter where the diagnosis was made: age, sex, chief complaint, history of present illness, a short assessment if available, and the reference acuity.

The context matters. Pregnancy codes (chapter O) take precedence for conditions in pregnancy, newborn codes (chapter P) apply to perinatal conditions in neonates, combination codes are preferred where they exist, post-MI complications have their own codes, and so on.

## Verdicts

- `correct`: the reference category is what a careful coder would assign here.
- `defensible`: a qualified coder could reasonably choose the reference category, even though another category might be better or equally good. This includes convention differences such as acute vs chronic variants, combination vs separate codes, an underlying cause vs a manifestation, or a reasonable reading of the chart.
- `wrong`: the reference category describes a different condition, or contradicts the name or the chart, so a qualified coder would not choose it. Made-up examples (not in your batch): "Acute appendicitis" coded as N39.0 (urinary tract infection, site not specified); "Cellulitis of the leg" coded as L40.0 (psoriasis vulgaris). Give the code you would assign in `correct_code`.
- `unsure`: you can't decide from the information given.

Be conservative: use `wrong` only when you're confident a qualified coder would reject the category. If the two patients would need different codes, judge the first patient and mention it in the reason.

## Output

Write one JSON object per item, one per line (JSON Lines), in the same order as the input, to the output file you're given:

{"id": "P0123", "verdict": "correct|defensible|wrong|unsure", "correct_code": "J95.84" or null, "confidence": "high|medium|low", "reason": "at most 25 words"}

`correct_code` is required for `wrong` and optional otherwise. Use real ICD-10-CM codes only.

## Rules

- Use only the item data and your own coding knowledge. Don't open any other file, don't search the web, and don't run any code except what's needed to write the output file.
- Judge every item in your batch. Don't skip any.
- When done, reply with only the count of each verdict and the output path.
