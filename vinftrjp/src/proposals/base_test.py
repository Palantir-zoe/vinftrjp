from src.proposals.base import Proposal

if __name__ == "__main__":
    p1 = Proposal()
    print(p1.idnamedict)
    print(p1.idpropdict)
    print()

    p2 = Proposal()
    print(p2.idnamedict)
    print(p2.idpropdict)
    print()

    print(Proposal.idnamedict)
    print(Proposal.idpropdict)
    print()

    # Set and Get Model Identifier
    p1.setModelIdentifier(1)
    print(p1.idnamedict)
    print(p1.idpropdict)
    print(p1.getModelIdentifier())
    print()
