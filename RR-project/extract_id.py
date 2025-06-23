import pandas as pd

# Example dataframe
df = pd.DataFrame({
    'info': ['aaabbbccddd(ABC12324)', 'xyz(testID456)', 'noidhere()', 'sample(withID999)']
})

# Extract content inside parentheses using regex
df['id'] = df['info'].str.extract(r'\(([^)]*)\)')

print(df)
