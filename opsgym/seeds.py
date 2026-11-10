"""Hand-written seeds. Phase 4 will generate more in this same shape.
The baseline task set lives in opsgym/tasks.py."""

SSO_SEED = {
    "now": "2026-10-29T10:00:00+00:00",  # a Thursday; "this week" = since Mon 2026-10-26
    "prompt": "Find which customer complained about SSO this week and open a P1 ticket "
              "for their account owner.",
    # Ground truth for verifiers. Never shown to the agent (env.reset only exposes prompt + now).
    "expected": {"account_id": "acc_acme", "priority": "P1", "owner_id": "u_priya"},
    "users": [
        {"id": "u_priya", "name": "Priya"},
        {"id": "u_dan", "name": "Dan"},
        {"id": "u_mei", "name": "Mei"},
    ],
    "accounts": [
        {"id": "acc_acme", "name": "Acme", "owner_id": "u_priya"},
        {"id": "acc_globex", "name": "Globex", "owner_id": "u_dan"},
        {"id": "acc_initech", "name": "Initech", "owner_id": "u_mei"},
    ],
    "messages": [
        # The real one: a complaint, this week.
        {"id": 1, "channel": "support", "account_id": "acc_acme", "author": "jo@acme.com",
         "text": "SSO login has been failing for our whole team since Monday.",
         "ts": "2026-10-27T15:12:00+00:00"},
        # Decoy by time: a real SSO complaint, but two weeks ago.
        {"id": 2, "channel": "support", "account_id": "acc_globex", "author": "al@globex.com",
         "text": "Our SSO redirect is broken, please help.",
         "ts": "2026-10-15T09:30:00+00:00"},
        # Decoy by sentiment: mentions SSO this week, but it's praise.
        {"id": 3, "channel": "support", "account_id": "acc_initech", "author": "bo@initech.com",
         "text": "Love the new SSO setup, rollout went great!",
         "ts": "2026-10-28T11:00:00+00:00"},
    ],
}


