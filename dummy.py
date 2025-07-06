Task 3: The Fare Investigator 💰
Goal: Explore how ticket prices affected survival chances

What to do:

Create a new feature called FarePerPerson:

df['FarePerPerson'] = df['Fare'] / df['FamilySize']
Create fare categories:

Free (fare = 0)
Cheap (fare: 0-10)
Moderate (fare: 10-50)
Expensive (fare: 50+)
Add these features to your model and test accuracy

Questions to answer:

Does fare per person predict survival better than total fare?
Were expensive tickets worth it for survival?
Why might some passengers have free tickets?
