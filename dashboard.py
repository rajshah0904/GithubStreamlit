import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('github_dataset.csv')
    data['language'].dropna()
    return data

# Normalize data and apply PCA to determine weights
def get_pca_weights(data):
    data_normalized = StandardScaler().fit_transform(data)
    pca = PCA(n_components=1)
    pca.fit(data_normalized)
    pca_loadings = pca.components_[0]
    return pca_loadings / np.sum(np.abs(pca_loadings))

# Calculate the Engagement Index
def calculate_engagement_index(data, weights):
    data['engagement_index'] = np.dot(data[['stars_count', 'forks_count', 'issues_count', 'pull_requests', 'contributors']], weights)
    data['engagement_index'] /= data['engagement_index'].max()
    return data

# Function to create visualizations
# Function to create enhanced visualizations with statistical insights
def create_visualizations(data):
    sns.set(style="whitegrid")

    # Visualization 1: Histogram of Engagement Index
    plt.figure(figsize=(12, 6))
    sns.histplot(data['engagement_index'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Engagement Index Across Repositories')
    plt.xlabel('Engagement Index')
    plt.ylabel('Number of Repositories')
    st.pyplot(plt)
    st.caption("""
    This histogram displays the distribution of the Engagement Index across various GitHub repositories. 
    It provides a visual representation of how engagement varies across the GitHub platform, 
    indicating the concentration of repositories at different levels of engagement. 
    The spread and peaks of the histogram reveal the commonality of engagement scores, 
    helping to identify whether most repositories have low, moderate, or high engagement. 
    This visualization is crucial in understanding the overall landscape of repository activities on GitHub.
    """)
    sns.set(style="whitegrid")

    # Visualization 2: Scatter Plot with Regression Line (Engagement Index vs. Stars)
    plt.figure(figsize=(12, 6))
    sns.regplot(x='engagement_index', y='stars_count', data=data, color='purple', scatter_kws={'alpha':0.5})
    plt.title('Engagement Index vs. Stars Count')
    plt.xlabel('Engagement Index')
    plt.ylabel('Stars Count')
    st.pyplot(plt)
    correlation_stars = np.corrcoef(data['engagement_index'], data['stars_count'])[0, 1]
    st.caption(f"""
    The scatter plot with a regression line demonstrates the relationship between the Engagement Index and the number of stars a repository has. 
    The correlation coefficient of {correlation_stars:.2f} provides a quantitative measure of this relationship. 
    A higher positive correlation suggests that repositories with more stars tend to have higher engagement. 
    This graph is pivotal in understanding how traditional measures of popularity (like stars) are aligned with the composite Engagement Index.
    """)

    # Visualization 3: Scatter Plot with Regression Line (Engagement Index vs. Forks)
    plt.figure(figsize=(12, 6))
    sns.regplot(x='engagement_index', y='forks_count', data=data, color='green', scatter_kws={'alpha':0.5})
    plt.title('Engagement Index vs. Forks Count')
    plt.xlabel('Engagement Index')
    plt.ylabel('Forks Count')
    st.pyplot(plt)
    correlation_forks = np.corrcoef(data['engagement_index'], data['forks_count'])[0, 1]
    st.caption(f"""
    This scatter plot, enhanced with a regression line, maps the Engagement Index against the number of forks for each repository. 
    With a correlation coefficient of {correlation_forks:.2f}, it offers insights into how collaborative activities, 
    represented by forks, correlate with the overall engagement. 
    This visualization is valuable for analyzing the impact of community contributions and collaboration on the engagement levels of repositories.
    """)

    # Visualization 4: Average Engagement Index by Top Languages
    plt.figure(figsize=(12, 6))
    top_languages = data.groupby('language')['engagement_index'].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=top_languages.index, y=top_languages.values, palette='muted')
    plt.title('Average Engagement Index by Top Programming Languages')
    plt.xlabel('Programming Language')
    plt.ylabel('Average Engagement Index')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.caption("""
    The bar chart ranks various programming languages based on their average Engagement Index, providing a glimpse into which languages are associated with highly active and engaging repositories. 
    By limiting the visualization to the top languages, it offers a focused view on where the most vibrant and dynamic communities exist within GitHub. 
    This insight is particularly useful for developers and organizations in making strategic decisions about technology adoption and community engagement.
    """)
    
    # Visualization 5: Bar Chart of Top 10 Users by Average Engagement Score
    plt.figure(figsize=(12, 6))
    data['username'] = data['repositories'].apply(lambda x: x.split('/')[0])
    avg_engagement_by_user = data.groupby('username')['engagement_index'].mean().sort_values(ascending=False)
    top_10_users = avg_engagement_by_user.head(10)
    sns.barplot(x=top_10_users.index, y=top_10_users.values, palette='viridis')
    plt.title('Top 10 Users by Average Engagement Score')
    plt.xlabel('Username')
    plt.ylabel('Average Engagement Score')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.caption("""
    This bar chart showcases the top 10 GitHub users in terms of average Engagement Index across their repositories. 
    It highlights individuals who consistently contribute to or maintain highly engaging content. 
    This can include a mix of popular projects (high star counts), active development (frequent pull requests), 
    and strong community involvement (many contributors). 
    Identifying these key influencers and active members provides valuable insights into the dynamics of collaboration and popularity in the GitHub ecosystem.
    """)
    
    # New Visualization 6: Box Plots for Issues and Pull Requests by Engagement Level
    data['engagement_category'] = pd.qcut(data['engagement_index'], 3, labels=["Low", "Medium", "High"])
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='engagement_category', y='issues_count', data=data)
    plt.title('Issues Count by Engagement Category')
    plt.xlabel('Engagement Category')
    plt.ylabel('Issues Count')
    plt.ylim(0, 200)

    plt.subplot(1, 2, 2)
    sns.boxplot(x='engagement_category', y='pull_requests', data=data)
    plt.title('Pull Requests by Engagement Category')
    plt.xlabel('Engagement Category')
    plt.ylabel('Pull Requests')
    plt.ylim(0, 200)
    
    st.pyplot(plt)
    st.caption("""
    These box plots illustrate the distribution of issues and pull requests across different engagement levels. 
    The plot for issues count shows the variation in the number of issues among repositories with varying engagement scores, 
    highlighting whether more engaged repositories tend to have more issues. 
    Similarly, the plot for pull requests provides insights into the level of active contributions and updates in repositories 
    across different engagement tiers. These visualizations are key to understanding the relationship between repository engagement 
    and the level of issue reporting and pull request activities, which are critical indicators of repository health and community involvement.
    """)

    # New Visualization 7: Scatter Plot - Contributor Count vs. Engagement Index
    plt.figure(figsize=(12, 6))
    sns.regplot(x='engagement_index', y='contributors', data=data, color='purple', scatter_kws={'alpha':0.5})
    plt.title('Engagement Index vs. Contributor Count')
    plt.xlabel('Engagement Index')
    plt.ylabel('Contributor Count')
    st.pyplot(plt)
    correlation_contributors = np.corrcoef(data['engagement_index'], data['contributors'])[0, 1]
    st.caption(f"""
        This scatter plot, with a regression line, examines the relationship between the Engagement Index and the number of contributors. 
        The correlation coefficient of {correlation_contributors:.2f} suggests the degree of linear relationship between these two metrics. 
        A higher correlation coefficient indicates a stronger relationship, where repositories with more contributors tend to have a higher Engagement Index. 
        This visualization helps in understanding the impact of community size and collaborative efforts on the overall engagement of GitHub repositories.
    """)


# Main function for Streamlit app
def main():
    st.title("GitHub Repositories Engagement Analysis")
    st.write("""
    Welcome to the GitHub Repositories Engagement Analysis App! This app delves into the fascinating world of GitHub repositories, 
    uncovering patterns of engagement and collaboration. Leveraging a unique Engagement Index, calculated through Principal Component Analysis (PCA), 
    this app evaluates repositories based on a blend of metrics: stars, forks, issues, pull requests, and contributor counts. 
    The visualizations aim to shed light on the dynamics of GitHub activity, from the popularity of programming languages to the influence of individual contributors.
    """)

    data = load_data()
    weights = get_pca_weights(data[['stars_count', 'forks_count', 'issues_count', 'pull_requests', 'contributors']])
    data = calculate_engagement_index(data, weights)
    
    st.header("Visualizations")
    st.write("The visualizations below provide insights into the engagement levels across different repositories and languages.")
    create_visualizations(data)
    st.header("Overall Insights")
    st.write("""
    The comprehensive visualizations presented in this app offer a deep dive into the dynamics of engagement across GitHub repositories, revealing several key insights:

    - **Engagement Index Distribution**: The histogram of the Engagement Index across repositories showcases a diverse landscape of engagement levels. It highlights that while a significant number of repositories have moderate engagement, there's a notable proportion with either very high or very low engagement. This variation points to the differing levels of activity and popularity among repositories on GitHub.

    - **Popularity vs. Engagement**: The scatter plots with regression lines comparing the Engagement Index to stars and forks counts demonstrate a positive correlation. This suggests that traditionally popular repositories (those with more stars and forks) also tend to have higher engagement. However, the correlation is not perfect, indicating other factors also contribute significantly to overall engagement.

    - **Community Involvement**: The analysis of issues and pull requests in relation to the Engagement Index, particularly the box plots, reveals that more engaged repositories tend to have a higher frequency of issues and pull requests. This underscores the importance of active participation and collaboration in driving repository engagement.

    - **Contributor Dynamics**: The scatter plot examining the relationship between contributor count and the Engagement Index, supported by the correlation coefficient, highlights that repositories with a larger community of contributors generally exhibit higher engagement. This correlation emphasizes the role of a robust contributing community in enhancing repository activity and engagement.

    - **Programming Languages and User Influence**: The bar charts showcasing the average Engagement Index by programming languages and the top users illustrate the influence of technology choices and individual contributors on engagement. Certain languages and users stand out for their association with highly engaged repositories, shedding light on trends and influential figures within the GitHub community.

    These insights collectively paint a picture of a vibrant and diverse GitHub ecosystem, where engagement is influenced by a mix of popularity, active development, community involvement, and individual contributions.
    """)

if __name__ == "__main__":
    main()
