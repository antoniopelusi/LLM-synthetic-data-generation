You are a synthetic data generator API. Output ONLY a valid JSON object.
No thinking, no reasoning, no explanation, no text before or after.
Any output outside the JSON block is a failure.
---USER---
Generate EXACTLY one NEW synthetic credit card transaction record.

# SCHEMA
| Field | Type | Constraints |
|-------|------|-------------|
| Unnamed: 0 | int | range [0, 1296674] |
| cc_num | int | range [60416207185, 4992346398065154048] |
| merchant | string | format: "fraud_[Surname] LLC", "fraud_[Surname] PLC", "fraud_[Surname] Ltd", "fraud_[Surname] Inc", "fraud_[Surname] Group", "fraud_[Surname]-[Surname]", "fraud_[Surname], [Surname] and [Surname]" |
| category | string | one of: gas_transport, grocery_pos, home, shopping_pos, kids_pets, shopping_net, entertainment, food_dining, personal_care, health_fitness, misc_pos, misc_net, grocery_net, travel |
| amt | float | range [1.00, 28948.90], 2 decimal places, heavily right-skewed: 25th pct=9.65, median=47.52, 75th pct=83.14, mean=70.35; most values between 5.00 and 150.00, rarely above 500.00 |
| first | string | common US first name |
| last | string | common US surname |
| gender | string | "F" or "M" |
| street | string | realistic US street address, formats: "123 Name St", "456 Name Ave Apt. 789", "789 Name Blvd Suite 012" |
| city | string | US city consistent with state |
| state | string | 2-letter US state code |
| zip | str | **string**, range [1257, 99783], consistent with state; low zip (East Coast) correlates with long near -70, high zip (West Coast) correlates with long near -120 |
| lat | float | range [20.027, 66.693], mean 38.538, consistent with city/state |
| long | float | range [-165.672, -67.950], mean -90.226, consistent with city/state, strongly negatively correlated with zip |
| city_pop | int | range [23, 2906700], heavily right-skewed: 25th pct=743, median=2456, 75th pct=20328 |
| job | string | realistic English job title |
| dob | string | format "YYYY-MM-DD", birth years 1940–1994 (cardholder must be at least 18 at time of transaction) |
| trans_num | string | exactly 32 lowercase hex characters, no dashes |
| unix_time | int | range [1325376018, 1371816817], median 1349249747 |
| merch_lat | float | near lat, difference typically < 1.0 degree |
| merch_long | float | near long, difference typically < 0.3 degrees |
| is_fraud | int | 0 with ~99.4% probability, 1 with ~0.6% probability |
| merch_zipcode | str | **string**, present ~84.9% of the time (**""** when absent, never null), range [1001, 99403], near zip |

# CONSTRAINTS
1. city, state, zip, lat, long must be geographically consistent with each other
2. merch_lat ≈ lat (difference < 1.0)
3. merch_long ≈ long (difference < 0.3)
4. merch_zipcode when not null must be near zip (within ±20000)
5. trans_num must be exactly 32 lowercase hex characters
6. zip and merch_zipcode MUST always be JSON strings. Output "" (empty string) for merch_zipcode when the value is absent; never use JSON null, the string "null", or NaN.
11. zip must be output as a zero-padded string if necessary (e.g. "01257" not 1257), matching real US ZIP format.
8. dob must produce an age ≥ 18 relative to the transaction date
9. Every field must be freshly generated, do NOT copy or reuse values from the examples
10. Randomize all fields: names, location, merchant, amount, date, trans_num must all be unique
11. Output must be strictly valid JSON: all numeric values must be plain numbers with no leading zeros, no underscores, no thousand separators, and no code-like syntax of any kind.
