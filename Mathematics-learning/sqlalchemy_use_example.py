from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()


class Account(Base):
    __tablename__ = u'acount'

    id = Column(Integer, primary_key=True)
    user_name = Column(String(50), nullable=False)
    password = Column(String(200), nullable=False)
    title = Column(String(50))
    salary = Column(Integer)

    def is_active(self):
        return True

    def get_id(self):
        return self.id

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False


import sqlalchemy as sa

print(sa.__version__)

conn = sa.create_engine('mysql://root:123456@127.0.0.1:3306/wikiurl')

metadata = sa.MetaData(conn)

# conn.execute("""CREATE TABLE zoo(criter VARCHAR(20) PRIMARY KEY,count INT, damages FLOAT)""")

ins = "INSERT INTO zoo(criter,count,damages) VALUES (?,?,?)"

# INSERT INTO zoo(criter,count,damages) VALUES ('bear',2,1000.0)

conn.execute(ins, 'bear', 2, 1000.0)
# conn.execute(ins, 'duck', 10, 0.0)
# conn.execute(ins, 'weasel', 1, 2000.0)
