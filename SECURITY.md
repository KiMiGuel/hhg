# Security

Thank you for helping keep HHG and its users safe.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security-sensitive bugs.**

Instead, email the maintainer at `security@hhg-project.example` (or open a
[GitHub private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
if the repo has them enabled).

Please include:

* A clear description of the issue and its impact.
* Reproduction steps (test image, command line, expected vs. actual).
* Whether you are OK with public disclosure after a fix is shipped.

We aim to acknowledge new reports within **3 business days** and ship a fix
or mitigation within **30 days** for confirmed high-severity issues.

## What HHG handles safely

* **API keys** – The pipeline reads `SERPAPI_KEY` from the environment
  (`.env` / shell). The `.env` file is **gitignored**. Never commit a key;
  rotate it if you accidentally push one.
* **Blockchain keys** – `PRIVATE_KEY` defaults to Anvil's well-known
  pre-funded test key.  It is a **test key only**.  For any non-local
  deployment, replace it with a dedicated testnet key and load it via
  the same `.env` mechanism.
* **No raw face crops persisted** – `temp/`, `cache/`, `reports/`, and the
  `data/captured_face.jpg`/`data/captured_from_url/` runtime crops are all
  gitignored.  The pipeline does not commit biometric data to the repo.
* **HTTPS-only outbound** – SerpApi, Wikipedia, profile pages, and the
  `catbox.moe` / `tmpfiles.org` image hosts are all reached over HTTPS.

## What you should **not** put in the repo

* Real `SERPAPI_KEY` values.  Use `.env` (gitignored) or your CI secrets.
* Real private keys for any chain that holds real assets.
* Face images of people who have not consented.  Public-figure press photos
  (e.g. `data/sample_face.jpg`) are acceptable for demo purposes.
* Output reports (`reports/`) and cache files (`cache/`) — already
  gitignored.

## Threat model (out of scope)

* **Private / login-gated platforms** – HHG only uses public image search.
  It cannot, and will not, bypass Instagram / Facebook / LinkedIn
  authentication, scrape private accounts, or break captchas.
* **Adversarial impersonation** – A live face matched by Lens is presented
  as a *candidate*, not proof.  The pipeline surfaces "No confident
  identification" on abstains and is not a forensic identification tool.
* **DDoS against SerpApi / catbox / Wikipedia** – HHG runs each search
  sequentially within a single call.  There is no fan-out beyond what the
  selected engine returns.

## Hardening checklist for deployments

* [ ] Rotate `SERPAPI_KEY`; do not reuse a dev key in production.
* [ ] Use a dedicated testnet wallet (never the Anvil default key).
* [ ] Set sensible timeouts (`SERPAPI_TIMEOUT_SECONDS`).
* [ ] Mount `temp/`, `cache/`, `reports/` on encrypted volumes if biometric
      data is sensitive.
* [ ] Strip `data/sample_face.jpg` if distributing to a pipeline that
      should not be associated with that individual.
