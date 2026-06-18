"""
Generates a synthetic, in-progress loan account for a newly registered user,
consistent with the rates/terms described in data.py (fixed ГЛП 40.55%, ГПР up
to 49.10%, €200-3000 principal, 3-18 month terms). Delegates all amortization
math to loan_service so seeded and "real" (applied-for) loans use identical
formulas.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict

from loan_service import ANNUAL_RATE, APR, PRINCIPAL_CHOICES, TERM_CHOICES, compute_amortization


def generate_loan_account(user_id: int) -> Dict[str, Any]:
    principal = random.choice(PRINCIPAL_CHOICES)
    term_months = random.choice(TERM_CHOICES)
    amort = compute_amortization(principal, term_months)

    installments_paid = random.randint(0, term_months - 1)
    outstanding_balance = round(amort["monthly_payment"] * (term_months - installments_paid), 2)

    return {
        "user_id": user_id,
        "status": "active",
        "principal_amount": round(principal, 2),
        "annual_rate": ANNUAL_RATE,
        "apr": APR,
        "term_months": term_months,
        "monthly_payment": amort["monthly_payment"],
        "total_payable": amort["total_payable"],
        "installments_paid": installments_paid,
        "outstanding_balance": outstanding_balance,
        "next_payment_date": dt.datetime.utcnow() + dt.timedelta(days=30),
    }
