library(dplyr)
library(tidyr)
library(ggplot2)
library(bsts)

df <- read.csv("local_data/data_cleaned_bt_split_subs_full_aug_annotated_dm_vdem_complete_notxt.csv")


############### PREPPING DATA ##################################################
# Check if the column has no null values
has_no_null <- !any(is.na(df$publish_date))

# Print the result
if (has_no_null) {
  cat("The column has no null values.")
} 


# Convert the 'date' column to a Date type
df$date <- as.Date(df$publish_date)

# Extract the month from the 'date' column
df$month <- format(df$date, "%Y-%m")

################################################################################


############### VISUALS ##################################################

ggplot(grouped, aes(x = month, y = count, group = 1)) +
  geom_line(aes(color = "Count"), size = 1.5) +
  geom_line(aes(y = sum_myth_count, color = "Sum Myth Count"), size = 1.5) +
  labs(title = "Count and Sum Myth Count by Month",
       x = "Month",
       y = "Value") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

################################################################################





# Replace NA values in 'myth_count' column with 0
df <- df %>%
  mutate(myth_count = replace_na(myth_count, 0))


# Group the dataframe by month and country, and calculate the count and sum
grouped <- df %>%
  group_by(month) %>%
  summarise(count = n(), sum_myth_count = sum(myth_count, na.rm = TRUE))

# Group the dataframe by month and calculate the count and sum, handling NA values
grouped <- df %>%
  group_by(month, country) %>%
  summarise(count = n())


# View the resulting dataframe
print(grouped, n=100)


# Create a state specification
ss <- AddLocalLinearTrend(list(), grouped$count)

# Add the regressor
ss <- AddRegressor(ss, grouped$sum_myth_count, standardize = TRUE)

# Fit the model
model <- bsts(df$sum_myth_count, state.specification = ss, niter = 500, ping = 0, seed = 123)

# Check the model
summary(model)



