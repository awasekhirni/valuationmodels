# Summary (ELI5)

# Imagine Anthropic is a lemonade stand that today sells $20 in lemonade but spends $27 on lemons and labour — it is losing money. You still believe the stand will become a giant chain. To decide what the stand is worth today, you write down one story: "How much lemonade will we sell each year? What profit per cup? How many new stands do we need? How risky is the bet?" You then convert every future dollar back to today's dollars (because money later is worth less than money now) and add it all up. That total — after subtracting debts and adding cash — is the value of the business. This module implements exactly that arithmetic in Python, with every number traceable to the story.



"""eli5_walkthrough.py — see the four levers in 20 lines."""
revenue_today = 100.0     # $100 of lemonade
growth        = [0.50, 0.30, 0.20]   # how fast it grows
margin        = 0.20      # profit per dollar of lemonade
wacc          = 0.10      # "money later is worth less" rate

print(f"Today: ${revenue_today}")
print(f"Story: grow {growth}, keep {margin:.0%} of sales, "
      f"discount at {wacc:.0%}\n")

total_value = 0.0
for t, g in enumerate(growth, start=1):
    revenue_today *= 1 + g
    profit_t = revenue_today * margin
    pv_t = profit_t / (1 + wacc) ** t
    total_value += pv_t
    print(f"Year {t}: sell ${revenue_today:6.1f}, "
          f"keep ${profit_t:5.1f}, today = ${pv_t:5.1f}")
print(f"\nValue today = ${total_value:,.2f}")
