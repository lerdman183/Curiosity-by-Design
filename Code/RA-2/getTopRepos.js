// Import required libraries
import { Octokit } from "@octokit/core";
import dotenv from "dotenv";
import fs from "fs/promises";

// Load environment variables
dotenv.config();

// Octokit initialization according to the documentation
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
  headers: {
    "X-GitHub-Api-Version": "2022-11-28",
  },
});

// Hardcoded list of top languages as per current TIOBE Index (Oct 2024)
const topLanguages = [
  "Python",
  "C",
  "C++",
  "Java",
  "C#",
  "JavaScript",
  "SQL",
  "PHP",
  "Go",
  "Assembly language",
  "MATLAB",
  "Fortran",
  "Kotlin",
  "Rust",
  "R",
  "Delphi/Object Pascal",
  "Swift",
  "Perl",
  "Scala",
  "COBOL",
  "D",
  "Dart",
  "Julia",
  "Lua",
  "Haskell",
  "VBScript",
  "Ada",
  "TypeScript",
  "Lisp",
  "Prolog",
  "F#",
  "Scratch",
  "Solidity",
  "Erlang",
  "Tcl",
  "VHDL",
  "ABAP",
  "Hack",
  "Logo",
  "ML",
];

// Function to fetch repos by language
async function fetchReposByLanguage(language, limit) {
  const response = await octokit.request("GET /search/repositories", {
    q: `language:${language}`,
    sort: "stars",
    order: "desc",
    per_page: limit,
  });
  return response.data.items.map((repo) => repo.html_url);
}

// Main function to orchestrate fetching and saving
async function main() {
  try {
    const repoUrls = {
      top10: [],
      rank11to20: [],
      rank21to30: [],
      rank31to40: [],
    };

    // Fetch repositories based on specified ranges and store them in appropriate categories
    for (let i = 0; i < topLanguages.length; i++) {
      const language = topLanguages[i];
      let limit;

      if (i < 10) limit = 100;
    //   else if (i < 20) limit = 75;
    //   else if (i < 30) limit = 50;
    //   else if (i < 40) limit = 25;
      else continue;

      console.log(`Fetching top ${limit} repos for ${language}...`);
      const repos = await fetchReposByLanguage(language, limit);

      // Store repos in corresponding rank category
      if (i < 10) repoUrls.top10.push(...repos);
    //   else if (i < 20) repoUrls.rank11to20.push(...repos);
    //   else if (i < 30) repoUrls.rank21to30.push(...repos);
    //   else repoUrls.rank31to40.push(...repos);
    }

    // Save results to a JSON file
    await fs.writeFile(
      "top_github_repos.json",
      JSON.stringify(repoUrls, null, 2)
    );
    console.log("Top repositories written to top_github_repos.json");
  } catch (error) {
    console.error("Error fetching repositories:", error);
  }
}

// Run main function
main();
