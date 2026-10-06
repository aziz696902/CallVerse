# Third-Party Notices

## HelpPilot

- **Project:** HelpPilot
- **Upstream repository:** https://github.com/poysa213/HelpPilot
- **Use in CallVerse:** Starting foundation for the Customer Advisor application
- **Declared license:** MIT License
- **Starting commit:** `3767824fb90b89a8b19fc4169d912645aaf6fe0b`

The upstream `LICENSE` file is preserved in this repository. Its copyright line
contains the literal placeholder `Copyright (c) 2026 <Your Name>`; CallVerse does not
invent or substitute an upstream author. This notice records provenance for academic
transparency and does not replace or modify the upstream license text.

## Calibration datasets

CallVerse uses aggregate statistics derived from the Technion Anonymous Bank
Call-Center Data (free-use statement with acknowledgement and notification requested)
and the Olist Brazilian E-Commerce Public Dataset (official Kaggle page reports
CC BY-NC-SA 4.0). Raw records are not tracked. See `docs/DATA_SOURCES.md` for official
links, permitted analytical roles, attribution details, and source hashes.

The Phase 5 classifier uses Bitext's Customer Support LLM Chatbot Training Dataset,
released on the official Bitext Hugging Face page under CDLA-Sharing-1.0. Only compact
derived evaluation artifacts are tracked; see `docs/DATA_SOURCES.md` for the exact
revision, file hash, schema, and analytical scope.

## Call-Center-Intelligence-System

- **Project:** Call Center Intelligence System
- **Upstream repository:** https://github.com/ANI-IN/Call-Center-Intelligence-System
- **Inspected revision:** `fed4b610742c1337147281fcec8f2cdcc0a79be5`
- **Copyright:** Copyright (c) 2026 Animesh Kumar
- **License:** MIT License
- **Use in CallVerse:** The Quality Analyst adapts the upstream patterns of bounded
  typed dimension scores, structured LLM output, explicit compliance flags,
  deterministic weighted-score recomputation, and manager-level quality summaries.

CallVerse implements its own delivery/e-commerce evidence contract, six-dimension
rubric, hard guardrails, benchmark, and aggregation code. It does not copy the
upstream audio/Whisper pipeline, Gradio UI, LangGraph workflow, database, report
generator, or security pipeline.

MIT License

Copyright (c) 2026 Animesh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Stable-Baselines3

- **Project:** Stable-Baselines3 2.9.0
- **Repository:** https://github.com/DLR-RM/stable-baselines3
- **Release revision:** `8908708f10c8ff29759c67f55c8acb56cab27463`
- **Copyright:** Copyright (c) 2019 Antonin Raffin
- **License:** MIT License
- **Use in CallVerse:** CPU PPO implementation for the isolated Phase 11 workforce
  policy experiment. Optional `extra` dependencies are not installed.

The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Gymnasium

- **Project:** Gymnasium 1.4.0
- **Repository:** https://github.com/Farama-Foundation/Gymnasium
- **Copyright:** Copyright (c) 2016 OpenAI; Copyright (c) 2022 Farama Foundation
- **License:** MIT License
- **Use in CallVerse:** Standard environment, observation-space, and action-space API
  used by the Phase 11 experiment.

The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Queueing-Simulation-and-Optimization-System

- **Project:** Call-Center Staffing Simulator (M/M/c)
- **Upstream repository:** https://github.com/thelostbong/Queueing-Simulation-and-Optimization-System
- **Adapted revision:** `762d29ee4aac62d5e184ad8a2b5f524bed60dcd5`
- **Copyright:** Copyright (c) 2026 Nayeemuddin Mohammed
- **License:** MIT License
- **Use in CallVerse:** Phase 10 adapts the M/M/c offered-load formulation,
  Erlang-C waiting probability and theoretical expected-wait structure, staffing
  sweeps, and minimum SLA staffing pattern. CallVerse replaces direct factorial
  calculations with a stable Erlang-B recurrence and adds its own typed contracts,
  calibrated AHT, service-level probability, occupancy constraint, forecast buffer,
  smoothing, shortfall handling, and Digital Twin validation.

CallVerse does not copy the upstream simulator, plotting application,
time-dependent simulator, generated figures, or its hardcoded €28/hour cost
assumption. The calibrated CallVerse SimPy Digital Twin remains authoritative.

MIT License

Copyright (c) 2026 Nayeemuddin Mohammed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
