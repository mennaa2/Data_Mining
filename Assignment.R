library(kknn)
library(ggplot2)
library(GGally)
library(dplyr)

# Part 1: Data Exploration & Preprocessing

# Load dataset
data <- read.csv("wdbc.data", header = FALSE)

# Assign column names
colnames(data) <- c("ID", "Diagnosis",
                    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
                    "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
                    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")

# Quick look at data
head(data)
str(data)
summary(data)
sum(is.na(data)) # Check missing values

# Class distribution
table(data$Diagnosis)

# Part 1.3: Visualizations

# Barplot of diagnosis distribution
diagnosis_counts <- table(data$Diagnosis)
barplot(diagnosis_counts,
        col=c("lightblue","pink"),
        main="Distribution of Diagnosis",
        xlab="Diagnosis",
        ylab="Count")

# Scatter plot: Radius vs Area colored by Diagnosis
plot(data$radius_mean, data$area_mean,
     col=ifelse(data$Diagnosis=="M","red","blue"),
     pch=19,
     xlab="Radius Mean",
     ylab="Area Mean",
     main="Radius vs Area by Diagnosis")
legend("topright",
       legend=c("Benign","Malignant"),
       col=c("blue","red"),
       pch=19)

# Histogram of Radius Mean
hist(data$radius_mean,
     main="Distribution of Radius Mean",
     xlab="Radius Mean",
     col="lightgreen")

# Boxplot of Radius Mean by Diagnosis
boxplot(radius_mean ~ Diagnosis,
        data=data,
        col=c("lightblue","pink"),
        main="Radius Mean by Diagnosis",
        xlab="Diagnosis",
        ylab="Radius Mean")

# Scatter: Area vs Perimeter
plot(data$area_mean, data$perimeter_mean,
     col=ifelse(data$Diagnosis=="M","red","blue"),
     pch=19,
     xlab="Area Mean",
     ylab="Perimeter Mean",
     main="Area vs Perimeter by Diagnosis")
legend("topright",
       legend=c("Benign","Malignant"),
       col=c("blue","red"),
       pch=19)

# Pairplot for top features
top_features <- c("radius_mean","area_mean","perimeter_mean",
                  "concavity_mean","concave_points_mean","Diagnosis")
data_top <- data[, top_features]
data_top$Diagnosis <- as.factor(data_top$Diagnosis)
ggpairs(data_top, aes(color=Diagnosis, alpha=0.6)) + 
  ggtitle("Pairplot of Top Features")

# Part 2: Model Preparation

# Convert Diagnosis to factor
data$Diagnosis <- as.factor(data$Diagnosis)

# Remove ID column
data <- data[,-1]

# Train-test split (70/30)
set.seed(123)
train_index <- sample(1:nrow(data), 0.7*nrow(data))
train_data <- data[train_index,]
test_data <- data[-train_index,]

cat("Training set:", nrow(train_data), "instances\n")
cat("Testing set:", nrow(test_data), "instances\n")

# Part 2.3: Train k-NN model

# Train k-NN with k=5 (common choice, balances bias/variance)
model <- kknn(Diagnosis ~ ., 
              train = train_data, 
              test = test_data, 
              k = 5)

# Predicted classes
predictions <- fitted(model)
head(predictions, 10)

# Part 3: Model Evaluation

# Confusion matrix
conf_matrix <- table(Actual=test_data$Diagnosis, Predicted=predictions)
conf_matrix

# Performance metrics
TN <- conf_matrix[1,1]
FP <- conf_matrix[1,2]
FN <- conf_matrix[2,1]
TP <- conf_matrix[2,2]

accuracy <- round((TP + TN)/sum(conf_matrix),4)
precision <- round(TP/(TP + FP),4)
recall <- round(TP/(TP + FN),4)
f1 <- round(2*(precision*recall)/(precision+recall),4)

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1, "\n")