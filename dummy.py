Task 4: The Title Extractor 🎭
Goal: Extract titles from passenger names and use them as features

What to do:

Extract titles from the Name column:

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
Group rare titles together:

# Keep common titles, group others as 'Other'
common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
Convert to dummy variables and add to your model

Questions to answer:

Which titles had the highest survival rates?
Did adding titles improve your model's accuracy?
What does "Master" mean, and why might it be important?
Hint: Master was used for young boys, Mrs for married women, Miss for unmarried women