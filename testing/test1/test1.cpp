#include <list>
#include <string>


/**
 * Define four types to represent the four different kinds of items
 * in our syntax tree. The base just contains a 'kind' field
 * that identifies the type of term.
 */

enum Kind {
    IDENTIFIER, CONSTANT, APPLICATION, ABSTRACTION
};

/**
 * Base term, derived types can be one of the four
 * types below.
 */
struct Term {
    Kind kind;
};

/**
 * An identifier or symbol.
 */
struct Identifier : Term {
    Identifier() : Term{IDENTIFIER} {};

    std::string id;
};

struct Constant : Term {
    Constant() : Term{CONSTANT} {};

    double val = 0;
};

typedef std::list<Term*> Terms;

struct Application : Term {
    Application() : Term{APPLICATION} {};

    Terms terms;
};

typedef std::list<Identifier*> Identifiers;

struct Abstraction : Term {
    Abstraction() : Term{ABSTRACTION} {};

    Identifiers args;
    Term *body = nullptr;
};

bool IsIsomorphic(Term *lhs, Term *rhs);


static bool IsIsomorphic_Term(Term *lhs, Identifiers &lhsAcc, Term *rhs, Identifiers& rhsAcc);

static bool IsIsomorphic_Application(Application *lhs, Identifiers &lhsAcc,
        Application *rhs, Identifiers& rhsAcc);

static bool IsIsomorphic_Identifier(Identifier *lhs, Identifiers &lhsAcc,
        Identifier *rhs, Identifiers& rhsAcc);

static bool IsIsomorphic_Constant(Constant *lhs, Constant *rhs);

static bool IsIsomorphic_Abstraction(Abstraction *lhs, Identifiers &lhsAcc,
        Abstraction *rhs, Identifiers& rhsAcc);


/**
 * A pair of Constants are isomorphic if they contain the same value.
 */
bool IsIsomorphic_Constant(Constant *lhs, Constant *rhs) {
    return lhs->val == rhs->val;
}

/**
 * Determines if two Application are isomorphic. This function
 * traverses the terms list of both applications in the exact same order, and
 * checks to see if each pair is isomorphic.
 *
 * Will return early at the first non-isomorphic pair.
 *
 * Because the loop is over both lists concurrently, the last statement
 * ensures that the same number of items were processed in both sides.
 */
bool IsIsomorphic_Application(Application *lhs, Identifiers &lhsAcc,
        Application *rhs, Identifiers& rhsAcc) {
    Terms::iterator i = lhs->terms.begin();
    Terms::iterator j = rhs->terms.begin();

    for(; i != lhs->terms.end() && j != rhs->terms.end(); ++i , ++j) {
        if (!IsIsomorphic_Term(*i, lhsAcc, *j, rhsAcc)) {
            return false;
        }
    }

    return i == lhs->terms.end() && j == rhs->terms.end();
}

/**
 * Search a list of identifiers for a particular identifier,
 * compare based on equality of the strings, return the position,
 * -1 if not found.
 */
static int identifierPos(const Identifiers& ids, const Identifier *id) {
    int pos = 0;
    for(auto i = ids.begin(); i != ids.end(); ++i, ++pos) {
        if((*i)->id == id->id) {
            return pos;
        }
    }
    return -1;
}

/**
 * Two identifiers are considered isomorphic is they appear
 * in the same position in the argument list of a function definition.
 *
 * For example, x,y`(+ x y) is isomorphic to `a,b`(+ a b) because
 * x appears in the 0'th position and y appears in the 1st position, just like
 * a appears in the 0'th position and b appears in the 1st position.
 *
 * This approach also handles nested lexically scoped function definitions, where
 * a variable might be defined in a higher level block. In this case, the
 * search function simply looks further back in the list. Variables are
 * matched for position on the first scoping block they are found in.
 */
bool IsIsomorphic_Identifier(Identifier *lhs, Identifiers &lhsAcc,
        Identifier *rhs, Identifiers& rhsAcc) {

    int lhsPos = identifierPos(lhsAcc, lhs);
    int rhsPos = identifierPos(rhsAcc, rhs);

    return lhsPos == rhsPos;
}

/**
 * Determine if two terms are isomorphic.
 *
 * The basic idea here is we perform a recursion in the same order down
 * both the left and right hand sides concurrently. Because the tree traversal
 * order is identical for both sides, we encounter each terms on both sides
 * in the exact same order. This lets us compare the structure of each term
 * and we don't have to find the position of each item. The item position
 * is not relevant because we perform the traversal in the same order.
 *
 * This first checks the kind of both the left and right hand sides, and if so,
 * calls one of the specialized functions for that type.
 *
 * This approach uses two accumulators, one for each side. The accumulators
 * hold a list of function definition arguments. This also lets us check
 * nested function definitions, as each level of nesting simply pushes those
 * arguments onto the accumulator.
 */
bool IsIsomorphic_Term(Term *lhs, Identifiers &lhsAcc, Term *rhs, Identifiers& rhsAcc) {

    if(lhs->kind == IDENTIFIER && rhs->kind == IDENTIFIER) {
        return IsIsomorphic_Identifier((Identifier*)lhs, lhsAcc, (Identifier*)rhs, rhsAcc);
    }

    else if(lhs->kind == CONSTANT && rhs->kind == CONSTANT) {
        return IsIsomorphic_Constant((Constant*)lhs, (Constant*)rhs);
    }

    else if(lhs->kind == APPLICATION && rhs->kind == APPLICATION) {
        return IsIsomorphic_Application((Application*)lhs, lhsAcc, (Application*)rhs, rhsAcc);
    }

    else if(lhs->kind == ABSTRACTION && rhs->kind == ABSTRACTION) {
        return IsIsomorphic_Abstraction((Abstraction*)lhs, lhsAcc, (Abstraction*)rhs, rhsAcc);
    }

    return false;
}


/**
 * Checks to see if two abstractions are isomorphic.
 *
 * The idea here is the two accumulators store the function definition arguments.
 * This pushes the arguments of the left and right hand side abstractions onto
 * their respective accumulators, and checks if the abstraction body
 * are isomorphic. Then it cleans up the accumulator by removing the
 * args that were pushed onto it.
 */
bool IsIsomorphic_Abstraction(Abstraction *lhs, Identifiers &lhsAcc,
        Abstraction *rhs, Identifiers& rhsAcc) {

    lhsAcc.insert(lhsAcc.begin(), lhs->args.begin(), lhs->args.end());
    rhsAcc.insert(rhsAcc.begin(), rhs->args.begin(), rhs->args.end());

    bool result = IsIsomorphic_Term(lhs->body, lhsAcc, rhs->body, rhsAcc);

    lhsAcc.erase(lhsAcc.begin(), std::next(lhsAcc.begin(), lhs->args.size()));
    rhsAcc.erase(rhsAcc.begin(), std::next(rhsAcc.begin(), rhs->args.size()));

    return result;
}

/**
 * The top level public IsIsomorphic function. This determines
 * if the structure of two terms is isomorphic.
 *
 * This function allocates two lists used as accumulators, and calls
 * the private accumulator IsIsomorphic_Term func.
 */
bool IsIsomorphic(Term *lhs, Term *rhs) {
    Identifiers lhsAcc;
    Identifiers rhsAcc;
    return IsIsomorphic_Term(lhs, lhsAcc, rhs, rhsAcc);
}



int main(int argc, const char** argv) {
    return 0;
}
