"""~20 hand-written baseline tasks in 3 families over one shared company world (Phase 3).
Each task = seed (world + messages + now + prompt + expected) plus a test-only `solution`
that tests/test_tasks.py replays through the tools to prove the task is solvable (D22).
The agent never sees `expected`, `solution` or `family` (env.reset exposes prompt + now only)."""

USERS = [{"id": f"u_{n.lower()}", "name": n} for n in ["Priya", "Dan", "Mei", "Omar", "Lena"]]
ACCOUNTS = [  # several owners have 2 accounts; "Acme" vs "Acme Labs" is deliberately confusable
    ("acc_acme", "Acme", "u_priya"), ("acc_acmelabs", "Acme Labs", "u_omar"),
    ("acc_globex", "Globex", "u_dan"), ("acc_initech", "Initech", "u_mei"),
    ("acc_umbrella", "Umbrella", "u_omar"), ("acc_hooli", "Hooli", "u_lena"),
    ("acc_stark", "Stark Industries", "u_dan"), ("acc_wayne", "Wayne Enterprises", "u_lena")]
OWNER = {a: o for a, _, o in ACCOUNTS}

A_PROMPT = ("Find which customer complained about {topic} this week (since Monday) and open a "
            "P1 ticket for their account owner.")
B_PROMPT = ("A customer reported a problem with {topic} in a support message today. Open a "
            "ticket for their account "
            "owner with the right priority: P1 if it is down for all of their users, P2 if only "
            "some users are affected, P3 for questions or minor issues.")


def _task(family, now, prompt, messages, account, priority, solution):
    msgs = [{"id": i, "channel": "support", "account_id": acc,
             "author": f"support@{acc.removeprefix('acc_')}.com", "text": text, "ts": ts}
            for i, (acc, ts, text) in enumerate(messages, start=1)]
    return {"family": family, "now": now, "prompt": prompt, "users": USERS,
            "accounts": [{"id": i, "name": n, "owner_id": o} for i, n, o in ACCOUNTS],
            "messages": msgs, "solution": solution,
            "expected": {"account_id": account, "priority": priority, "owner_id": OWNER[account]}}


def complaint(topic, now, target, decoy_old, decoy_praise, noise):
    """Family A: real complaint this week + same-topic complaint before Monday (decoy by time)
    + same-topic praise this week (decoy by sentiment) + other-topic complaint (noise).
    Each message arg = (account_id, ts, text); the target is always message 1."""
    return _task("complaint_this_week", now, A_PROMPT.format(topic=topic),
                 [target, decoy_old, decoy_praise, noise], target[0], "P1",
                 {"query": topic, "message_id": 1})


def triage(topic, now, priority, target, noise):
    """Family B, the HELD-OUT family (D25): priority must be inferred from the severity in
    today's message. No same-topic time decoy (removed in D25), so B isolates one skill,
    severity -> priority, plus the owner lookup. Its natural shortcut is omitting priority
    (default P3), a different corner from family C's unassigned owner."""
    return _task("severity_triage", now, B_PROMPT.format(topic=topic),
                 [target, noise], target[0], priority, {"query": topic, "message_id": 1})


def named(account, prompt, priority):
    """Family C: the prompt names the account; owner must still be looked up."""
    name = next(n for i, n, _ in ACCOUNTS if i == account)
    return _task("named_account", "2026-10-30T10:00:00+00:00", prompt, [], account, priority,
                 {"account_query": name})


TASKS = {
    # ---- A: complaint this week (P1) ----
    "a_sso": complaint("SSO", "2026-10-29T10:00:00+00:00",
        ("acc_acme", "2026-10-27T15:12:00+00:00", "SSO login has been failing for our whole team since Monday."),
        ("acc_globex", "2026-10-15T09:30:00+00:00", "Our SSO redirect is broken, please help."),
        ("acc_initech", "2026-10-28T11:00:00+00:00", "Love the new SSO setup, rollout went great!"),
        ("acc_hooli", "2026-10-28T09:00:00+00:00", "Invoices are showing the wrong currency.")),
    "a_billing": complaint("billing", "2026-10-21T16:00:00+00:00",
        ("acc_globex", "2026-10-20T10:00:00+00:00", "We were double charged, billing is a mess this month."),
        ("acc_hooli", "2026-10-16T14:00:00+00:00", "Billing charged us twice, please fix."),  # last Fri: inside 7 days, before Monday
        ("acc_stark", "2026-10-21T09:00:00+00:00", "Thanks, the new billing page is much clearer."),
        ("acc_initech", "2026-10-19T11:00:00+00:00", "API calls are timing out.")),
    "a_export": complaint("export", "2026-10-16T09:00:00+00:00",
        ("acc_initech", "2026-10-14T13:00:00+00:00", "CSV export keeps failing with a server error."),
        ("acc_umbrella", "2026-10-08T10:00:00+00:00", "The export button does nothing for us."),
        ("acc_wayne", "2026-10-15T08:00:00+00:00", "Export to CSV is fast now, nice work."),
        ("acc_acme", "2026-10-13T16:00:00+00:00", "Can't upload attachments larger than 5MB.")),
    "a_webhook": complaint("webhook", "2026-11-03T12:00:00+00:00",
        ("acc_umbrella", "2026-11-02T10:30:00+00:00", "Our webhook deliveries stopped completely, nothing arrives."),
        ("acc_initech", "2026-10-30T15:00:00+00:00", "Webhook retries are broken for us."),  # last Fri
        ("acc_globex", "2026-11-03T08:00:00+00:00", "The webhook logs feature is great, thanks!"),
        ("acc_stark", "2026-11-02T09:00:00+00:00", "Dashboard charts render blank.")),
    "a_password": complaint("password reset", "2026-10-09T15:00:00+00:00",
        ("acc_hooli", "2026-10-07T11:00:00+00:00", "Password reset emails never arrive for our users."),
        ("acc_acme", "2026-09-29T10:00:00+00:00", "Password reset link is expired when we click it."),
        ("acc_globex", "2026-10-08T12:00:00+00:00", "The new password reset flow is so smooth."),
        ("acc_wayne", "2026-10-06T09:00:00+00:00", "Search results are missing recent tickets.")),
    "a_ratelimit": complaint("rate limit", "2026-10-28T11:00:00+00:00",
        ("acc_stark", "2026-10-26T09:15:00+00:00", "We keep hitting the API rate limit and our sync is failing."),
        ("acc_umbrella", "2026-10-22T10:00:00+00:00", "Rate limit errors are blocking our imports."),
        ("acc_hooli", "2026-10-27T14:00:00+00:00", "Appreciate the higher rate limit on the new plan!"),
        ("acc_acmelabs", "2026-10-27T10:00:00+00:00", "Our SAML certificate expired, can't log in.")),
    "a_mobile": complaint("mobile app", "2026-11-05T10:00:00+00:00",
        ("acc_acmelabs", "2026-11-04T09:00:00+00:00", "The mobile app crashes on launch for everyone at Acme Labs."),
        ("acc_acme", "2026-10-28T10:00:00+00:00", "Mobile app crashes when opening tickets."),  # Acme vs Acme Labs
        ("acc_globex", "2026-11-03T11:00:00+00:00", "The mobile app redesign looks fantastic."),
        ("acc_initech", "2026-11-04T15:00:00+00:00", "Exports are missing the date column.")),
    # ---- B: severity triage (priority from the message) ----
    "b_dashboards": triage("dashboards", "2026-10-22T15:00:00+00:00", "P1",
        ("acc_wayne", "2026-10-22T09:00:00+00:00", "Dashboards are completely down for our whole company since this morning."),
        ("acc_acme", "2026-10-22T11:00:00+00:00", "Quick question about invoice dates.")),
    "b_search": triage("search", "2026-10-27T14:00:00+00:00", "P2",
        ("acc_globex", "2026-10-27T10:00:00+00:00", "Search is failing for some of our users; others are fine."),
        ("acc_hooli", "2026-10-27T12:00:00+00:00", "Can we change the color of the sidebar?")),
    "b_notifications": triage("notifications", "2026-11-04T11:00:00+00:00", "P1",
        ("acc_initech", "2026-11-04T08:30:00+00:00", "No one on our team is receiving notifications at all."),
        ("acc_stark", "2026-11-04T09:00:00+00:00", "Where do I download last month's invoice?")),
    "b_reporting": triage("reporting", "2026-10-14T16:00:00+00:00", "P2",
        ("acc_umbrella", "2026-10-14T10:00:00+00:00", "A few of our analysts can't open the reporting page; the rest of the team can."),
        ("acc_wayne", "2026-10-14T12:00:00+00:00", "Single sign-on is working great.")),
    "b_integration": triage("integration", "2026-10-30T10:00:00+00:00", "P3",
        ("acc_acme", "2026-10-30T08:00:00+00:00", "Quick question: does the Salesforce integration support custom fields?"),
        ("acc_globex", "2026-10-30T09:00:00+00:00", "Billing page shows an old address.")),
    "b_auditlog": triage("audit log", "2026-11-06T09:30:00+00:00", "P2",
        ("acc_hooli", "2026-11-06T08:00:00+00:00", "The audit log page errors for some admins but works for others."),
        ("acc_umbrella", "2026-11-06T07:00:00+00:00", "How do I add a new teammate?")),
    # ---- C: named account (owner lookup required; confusable names) ----
    "c_acme_renewal": named("acc_acme", "Acme's account owner needs to follow up on their renewal. "
                            "Open a P2 ticket for them.", "P2"),
    "c_acmelabs_contract": named("acc_acmelabs", "Acme Labs' account owner needs to review their "
                                 "new contract. Open a P2 ticket for them.", "P2"),
    "c_globex_onboarding": named("acc_globex", "Globex's account owner should schedule onboarding "
                                 "for their new team. Open a P2 ticket for them.", "P2"),
    "c_wayne_security": named("acc_wayne", "Wayne Enterprises' account owner must complete a "
                              "security review this week. Open a P1 ticket for them.", "P1"),
    "c_umbrella_invoice": named("acc_umbrella", "Umbrella's account owner needs to resolve an "
                                "invoice dispute. Open a P2 ticket for them.", "P2"),
    "c_initech_migration": named("acc_initech", "Initech's account owner should plan their data "
                                 "migration. Open a P2 ticket for them.", "P2"),
    "c_hooli_expansion": named("acc_hooli", "Hooli's account owner should prepare an expansion "
                               "proposal. Open a P1 ticket for them.", "P1"),
}
