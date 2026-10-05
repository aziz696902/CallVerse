# CallVerse Data Sources

## Technion Anonymous Bank Call-Center Data

- **Source:** Technion – Israel Institute of Technology, Service Enterprise Engineering Center
- **Period:** January–December 1999 (12 monthly call-level files)
- **Official documentation:** https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.pdf
- **Official archive directory:** https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/
- **Use statement:** the documentation says the data is free for use and asks users to acknowledge the source and notify Avi Mandelbaum. CallVerse has not sent any notification; the project owner must address this before publication or submission.
- **CallVerse use:** transferable arrival shape, generic service duration, queue waiting, abandonment, and right-censored patience behavior.
- **Not used for:** delivery behavior, e-commerce intents, personas, order status, refunds, or delivery policy. Bank service types and priority are never mapped to CallVerse intents or premium personas.

The archive used for this build had SHA-256 `110527823A296E3163566F5D10D687E13859C60D963FD403D4623D89C7CADC4B`.

## Brazilian E-Commerce Public Dataset by Olist

- **Source:** Olist, distributed through the official Kaggle dataset page
- **Period/scope:** approximately 100,000 Brazilian marketplace orders from 2016–2018 across nine source tables
- **Official source:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Reported license:** CC BY-NC-SA 4.0
- **CallVerse use:** the orders and reviews tables only, for promised-versus-actual delivery timing, descriptive lateness severity, and order-level review-score association.
- **Not used for:** call-center arrivals, queue mechanics, support-channel choice, contact probability, complaint probability, or refund-contact probability.

The download archive used for this build had SHA-256 `967E41E04FC306FE604E2A693F488995A8B41E5047418F8A5C8E4ABD6DECA784`.

The sources describe different organizations and people. They are calibrated independently and are never row-wise merged. Only small aggregate artifacts are tracked.
