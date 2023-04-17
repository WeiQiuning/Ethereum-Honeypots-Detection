from all_tables import Base
import config


def main():
    sqlalchemy_engine = config.create_sqlalchemy_engine()
    Base.metadata.create_all(sqlalchemy_engine)


if __name__ == '__main__':
    main()
