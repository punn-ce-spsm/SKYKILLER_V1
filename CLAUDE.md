# General Instructions

- Implement things in small chunks and verify in small chunks. Do not try to do everything at the same time.
- Always try to leverage MCP connectors, subagents, skills to your advantage.
- Never execute a vague task, always ask user to clarify the task until the task and context is crystal clear first.
- When done with any task, always use /rubberduck to see if the explanation makes sense and if it doesn't, iterate until it makes sense

# Engineering Instructions

- Avoid duplicates, try to keep lines of code at a minimum, write optimized code, think before you code.
- For ui, always do browser verification if possible.
- Before doing anything always check the latest documentation to make sure you implement everything corectly and efficiently.
- Save to git for every milestone reached.

# Frontend Instructions

- People hate ai generated things, all ui should never look ai generated/vibe coded.
- Keep ux as frictionless as possible, when verifying ux, use the user-shoes skill to go through the frontend like a real user and iterate until user-shoes pass
- When given a figma reference from the user, always try to implment it pixel perfect exactly but if implementing exactly doesn't make sense, ask user questions

# Documentation Instructions

- When anything is done, document the current project state in @MEMORY.md
- When any action can only be done by a human, document the exact instructions on what to do and clearly how to do in @ACTION.md
