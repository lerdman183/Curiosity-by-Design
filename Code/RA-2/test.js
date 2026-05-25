const getRepoInfoFromFile = (filePath) => {
  const repoUrls = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  // Access the top10 array
  const top10Repos = repoUrls.top10;

  // Loop through the top10 repos to process them
  top10Repos.forEach((repoUrl) => {
    const urlParts = repoUrl.split("/");
    const owner = urlParts[3]; // The correct index for the owner
    const repo = urlParts[4]; // The correct index for the repository name
    console.log(owner, repo); // Add logs to verify correct parsing
    return { owner, repo };
  });
};
