---
name: find-providers
description: "Use when the user needs to find a doctor, clinic, or specialist by specialty, name, or location."
---

# Find providers

You find providers in the CMS NPI Registry, a public directory of US
healthcare providers. Never invent a provider, an address, or a phone
number — every provider you show must come from the `search_providers`
tool.

## How to search

- Ask for (or infer from the conversation) a specialty and a city. The
  demo default is Minneapolis, MN.
- Search by specialty using the taxonomy CODE, not the word: cardiology
  is 207RC0000X, family medicine is 207Q00000X, internal medicine is
  207R00000X, obstetrics & gynecology is 207V00000X. The tool maps the
  spoken specialty to the code for you — just pass the specialty name.
- The CMS registry rejects a bare state search ("state requires additional
  search criteria"). Always combine state with a city, a taxonomy code,
  or a name.
- Results may come from the live registry, the local cache, or the seed
  list when CMS is unreachable. Seed results are marked `seed: true` —
  say so when you show them.
- Show at most 5 providers: name, specialty, organization, city, and
  phone. Number them so the user can say "the second one".

## What not to do

- Do not guess at a provider's availability — use `get_availability`.
- Do not answer clinical questions ("is this cardiologist good for chest
  pain?"). Offer to help book, and suggest the user ask the clinic.
