import { Octokit } from "@octokit/core";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pLimit from "p-limit";

// Manually define __dirname in ES module context
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config();

// Initialize Octokit with authentication token
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
});

// Initialize pLimit to limit the concurrency of API requests
const limit = pLimit(5); // Adjust this value based on your needs

// Read repository URLs from JSON file
const getRepoInfoFromFile = (filePath) => {
  const repoUrls = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  return repoUrls.top10; // Only process top10 repositories for now
};

// Fetch repository details to get the language
const fetchRepoLanguage = async (owner, repo) => {
  try {
    const { data: repoDetails } = await octokit.request(
      "GET /repos/{owner}/{repo}",
      {
        owner,
        repo,
      }
    );
    return repoDetails.language || "Unknown";
  } catch (error) {
    console.error(`Error fetching language for ${owner}/${repo}:`, error);
    return "Unknown";
  }
};

const readJsonFileSafely = (filePath) => {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  const fileContent = fs.readFileSync(filePath, "utf-8").trim();
  return fileContent ? JSON.parse(fileContent) : {};
};

const fetchRepoIssuesWithComments = async (repoUrl, totalIssuesCounter) => {
  const urlParts = repoUrl.split("/");
  const owner = urlParts[3];
  const repo = urlParts[4];

  try {
    console.log(`Fetching issues for ${owner}/${repo}`);

    const repoLanguage = await fetchRepoLanguage(owner, repo);

    const { data: issues } = await octokit.request(
      "GET /repos/{owner}/{repo}/issues",
      {
        owner,
        repo,
        per_page: 15,
        sort: "comments",
        direction: "desc",
      }
    );

    if (!Array.isArray(issues)) {
      throw new Error("Invalid issues response: Not an array");
    }

    const issuesWithComments = await Promise.all(
      issues.map(async (issue) => {
        const { data: comments } = await octokit.request(issue.comments_url);

        if (!Array.isArray(comments)) {
          throw new Error("Invalid comments response: Not an array");
        }

        const commentsWithReactions = await Promise.all(
          comments.slice(0, 7).map(async (comment) => {
            const { data: reactions } = await octokit.request(
              "GET /repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
              {
                owner,
                repo,
                comment_id: comment.id,
              }
            );

            return {
              comment_url: comment.html_url,
              user: comment.user.login,
              body: comment.body,
              created_at: comment.created_at,
              reactions: reactions.map((reaction) => ({
                content: reaction.content,
                count: reaction.count,
              })),
            };
          })
        );

        return {
          issue_url: issue.html_url,
          issue_body: issue.body,
          language: repoLanguage,
          comments: commentsWithReactions,
        };
      })
    );

    const outputFilePath = "issuesWithComments.json";
    let allRepoIssues = readJsonFileSafely(outputFilePath);
    allRepoIssues[repoUrl] = issuesWithComments;

    fs.writeFileSync(outputFilePath, JSON.stringify(allRepoIssues, null, 2));
    totalIssuesCounter.count += issues.length;

    console.log(`Issues with comments saved for ${owner}/${repo}`);
  } catch (error) {
    console.error(`Error fetching issues for ${owner}/${repo}:`, error);

    const failedReposFilePath = "failedRepos.json";
    let failedRepos = readJsonFileSafely(failedReposFilePath);

    if (!failedRepos.top10) {
      failedRepos.top10 = [];
    }
    failedRepos.top10.push(repoUrl);

    fs.writeFileSync(failedReposFilePath, JSON.stringify(failedRepos, null, 2));
    console.log("Rate limit hit or error occurred, waiting for 3 minutes...");
    await new Promise((resolve) => setTimeout(resolve, 180000));
  }
};

// Main function to control the flow
const main = async () => {
  const repoUrls = getRepoInfoFromFile(
    path.join(__dirname, "top_github_repos.json")
  );
  const totalIssuesCounter = { count: 0 };

  for (const repoUrl of repoUrls) {
    await limit(() => fetchRepoIssuesWithComments(repoUrl, totalIssuesCounter));
  }

  console.log(`Total issues collected: ${totalIssuesCounter.count}`);
};

// Start the process
main();