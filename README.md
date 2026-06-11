# publim

# Probability Uniqueness-Based Linkage Method (PUBLIM)
 - Developed for the NIH/NIA LINKAGE program
 - Links records without a universal identifier
 - Key innovation: uses rarity (uniqueness) of matching attributes
 - Rare matches provide stronger evidence than common matches

# Why
 - Traditional methods treat all exact matches similarly
  -- John Smith match ≠ Krzysztof Nowak match
 - PUBLIM measures how common a value is in Dataset B
 - More unique attributes receive higher linkage weight
 - Improves linkage precision and transparency

# Three Steps
- Step 1: Define variable categories and match-quality bands
- 1. Name, DOB, ZIP, Gender, Admin IDs
- Step 2: Generate candidate pools using strict-to-relaxed matching
- Step 3: Compute uniqueness score and select best candidate
- Assign Strong / Fair / Weak confidence levels

# Scoring Logic
 1. For each field compute pk = frequency of matching value
 2. Combined probability: P = product(pk)
 - Rare combinations produce very small P
 - Convert to log-odds score
 - Higher score = stronger evidence of true match

 # Reference:
 "Description of Matching Method for Linking Datasets in LINKAGE" by Thomas E MaCurdy
