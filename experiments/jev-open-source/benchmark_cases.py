"""Shared classification fixtures; no model imports or inference side effects.

Version 2 uses the original Laya examples and criterion descriptions for both
runners. Historical metask logs used different descriptions and five different
bilingual cases; do not compare their accuracy as an identical-input benchmark.
"""
BENCHMARK_VERSION = "ticket-classification-v2"
INSTRUCTIONS = "Which category does this message belong to?"

CASES = [
    # Billing and refunds
    (
        "billing",
        "I was charged twice for my subscription this month, please refund the duplicate.",
        "我这个月的订阅被扣了两次款，请把重复的那笔退给我。",
    ),
    (
        "billing",
        "The invoice shows an amount I did not agree to, I want it corrected.",
        "发票上的金额不是我认可的数额，我希望改正。",
    ),
    (
        "billing",
        "Can you switch me to the annual plan and refund the difference?",
        "能帮我换成年付套餐并退还差价吗？",
    ),
    ("billing", "My card was declined but the payment still went through.", "我的卡显示被拒，但钱还是扣走了。"),
    ("billing", "I need a VAT invoice for last quarter's payments.", "我需要上个季度付款的增值税发票。"),
    # Technical failures
    ("technical", "The app crashes every time I open the export dialog.", "每次打开导出对话框应用就崩溃。"),
    (
        "technical",
        "Webhook deliveries stopped arriving after yesterday's update.",
        "昨天的更新之后 webhook 推送就不再到达了。",
    ),
    (
        "technical",
        "I get a 500 error when uploading files larger than 20MB.",
        "上传超过 20MB 的文件时报 500 错误。",
    ),
    ("technical", "Two-factor codes are rejected even though they are correct.", "两步验证码是对的却一直被拒绝。"),
    ("technical", "The dashboard shows no data since Monday morning.", "周一早上开始仪表盘就不显示数据了。"),
    # Sales inquiries
    ("sales", "Do you offer discounts for non-profit organizations?", "非营利组织有折扣吗？"),
    (
        "sales",
        "What is the price difference between the Pro and Team tiers?",
        "Pro 版和 Team 版的价格差是多少？",
    ),
    ("sales", "Is there an education plan for university labs?", "大学实验室有教育版套餐吗？"),
    ("sales", "Can I get a quote for two hundred seats?", "两百个席位能给我一个报价吗？"),
    ("sales", "Do you have a reseller program in Southeast Asia?", "东南亚有代理商计划吗？"),
    # Other requests
    ("other", "Just wanted to say the team is doing excellent work.", "你们的团队做得很好，致谢。"),
    ("other", "Congratulations on the launch, looking forward to the roadmap.", "恭喜发布，期待后续路线图。"),
    ("other", "Is your office hiring for summer internships?", "你们办公室招暑期实习生吗？"),
    ("other", "I liked the talk your founder gave last week.", "我喜欢上周你们创始人的演讲。"),
    ("other", "Please add my name to the newsletter list.", "请把我加入邮件订阅名单。"),
]

CRITERIA = {
    "billing": "invoices, charges, refunds, payment methods",
    "technical": "bugs, outages, errors, integrations not working",
    "sales": "pricing, discounts, quotes, plans, reselling",
    "other": "praise, greetings, hiring, unrelated chat",
}
