import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/tickets_cleaned.csv')

# Basic info
print(f"Total tickets: {len(df)}")
print(f"\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nPriority distribution:")
print(df['priority'].value_counts())
print(f"\nStatus distribution:")
print(df['status'].value_counts())

# Chart 1: Category Distribution
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Ticket Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('screenshots/category_distribution.png')
print("\nChart 1 saved: screenshots/category_distribution.png")

# Chart 2: Priority Distribution
plt.figure(figsize=(10, 6))
df['priority'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Ticket Priority')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('screenshots/priority_distribution.png')
print("Chart 2 saved: screenshots/priority_distribution.png")

plt.close('all')