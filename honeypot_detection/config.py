import logging
import sqlalchemy

import etherscan


logging.basicConfig(level="INFO")


def create_etherscan_client():
    return etherscan.Client("RPTRFPYPP1P7XCXJJA3MHBHAM278EIZ7KP")


def create_sqlalchemy_engine():
    return sqlalchemy.create_engine("sqlite:///test.db") #honeypot-detection.db")
