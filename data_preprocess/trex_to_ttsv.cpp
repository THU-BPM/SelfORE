#include <algorithm>
#include <cstddef>
#include <codecvt>
#include <locale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"

using namespace rapidjson;
struct Handler: BaseReaderHandler<UTF8<>, Handler>{
	std::ostream &out;
	bool spoonly;
	Handler(std::ostream &out, bool spoonly): out(out), spoonly(spoonly) {}

	enum class State{
		INIT,
		DOCINTER,
		DOC,
		DOCID,
		TEXT,

		// Word boundaries
		WBINIT,
		WBINTER,
		WBFIRST,
		WBSECOND,
		WBEND,

		// Sentence boundaries
		SBINIT,
		SBINTER,
		SBFIRST,
		SBSECOND,
		SBEND,

		// Entities
		ENTINIT,
		ENTINTER,
		ENT,

		// Triples
		TRPINIT,
		TRPINTER,
		TRP,
		TRPSENID,
		TRPANN,

		TRPPRED,
		TRPPREDURI,
		TRPPREDANN,

		TRPSUB,
		TRPSUBURI,
		TRPSUBANN,
		TRPSUBBFIRST,
		TRPSUBBSECOND,

		TRPOBJ,
		TRPOBJURI,
		TRPOBJANN,
		TRPOBJBFIRST,
		TRPOBJBSECOND,
	} state;

	bool StartObject();
	bool EndObject(SizeType);
	bool StartArray();
	bool EndArray(SizeType);
	bool Key(char const*,SizeType,bool);
	bool String(char const*,SizeType,bool);
	bool Uint(unsigned);
	bool Null();

	void write_doc();

	unsigned docid;
	char text[1<<16];
	std::size_t text_size;
	std::vector<std::tuple<std::size_t, std::size_t>> words_boundaries, sentences_boundaries;

			// words_boundaries.push_back({x, 0});

	struct Triple{
		unsigned sentence;
		std::tuple<std::size_t, std::size_t> subject_boundaries, object_boundaries;
		unsigned subject, predicate, object;
	};
	std::vector<Triple> triples;
	bool ignore_triple=false;

	std::size_t nb_doc=0, nb_triple=0;
};

bool Handler::StartObject(){
	switch(state){
		case State::DOCINTER:
			words_boundaries.clear();
			sentences_boundaries.clear();
			triples.clear();
			state=State::DOC;
			return true;
		case State::ENTINTER:
			state=State::ENT;
			return true;
		case State::TRPINTER:
			if(!ignore_triple)
				triples.emplace_back();
			ignore_triple=false;
			state=State::TRP;
			return true;
	}
	return true;
}

bool Handler::EndObject(SizeType){
	switch(state){
		case State::ENT:     state=State::ENTINTER; return true;
		case State::TRP:     state=State::TRPINTER; return true;
		case State::TRPPRED: case State::TRPSUB: case State::TRPOBJ:
		                     state=State::TRP;      return true;
		case State::DOC:
			write_doc();
			state=State::DOCINTER;
			return true;
	}
	return true;
}

bool Handler::StartArray(){
	switch(state){
		case State::INIT:      state=State::DOCINTER;  return true;
		case State::WBINIT:    state=State::WBINTER;   return true;
		case State::WBINTER:   state=State::WBFIRST;   return true;
		case State::SBINIT:    state=State::SBINTER;   return true;
		case State::SBINTER:   state=State::SBFIRST;   return true;
		case State::ENTINIT:   state=State::ENTINTER;  return true;
		case State::TRPINIT:   state=State::TRPINTER;  return true;
	}
	return true;
}

bool Handler::EndArray(SizeType){
	switch(state){
		case State::DOCINTER:  state=State::INIT;      return true;
		case State::WBINTER:   state=State::DOC;       return true;
		case State::WBEND:     state=State::WBINTER;   return true;
		case State::SBINTER:   state=State::DOC;       return true;
		case State::SBEND:     state=State::SBINTER;   return true;
		case State::ENTINTER:  state=State::DOC;       return true;
		case State::TRPINTER:
			if(ignore_triple)
				triples.pop_back();
			ignore_triple=false;
			state=State::DOC;
			return true;
	}
	return true;
}

bool Handler::Key(char const *str, SizeType, bool){
	switch(state){
		case State::DOC:
			if(!std::strcmp(str, "docid"))
				state=State::DOCID;
			else if(!std::strcmp(str, "text"))
				state=State::TEXT;
			else if(!std::strcmp(str, "words_boundaries"))
				state=State::WBINIT;
			else if(!std::strcmp(str, "sentences_boundaries"))
				state=State::SBINIT;
			else if(!std::strcmp(str, "entities"))
				state=State::ENTINIT;
			else if(!std::strcmp(str, "triples"))
				state=State::TRPINIT;
			return true;
		case State::TRP:
			if(!std::strcmp(str, "predicate"))
				state=State::TRPPRED;
			else if(!std::strcmp(str, "subject"))
				state=State::TRPSUB;
			else if(!std::strcmp(str, "object"))
				state=State::TRPOBJ;
			else if(!std::strcmp(str, "sentence_id"))
				state=State::TRPSENID;
			else if(!std::strcmp(str, "annotator"))
				state=State::TRPANN;
			return true;
		case State::TRPPRED:
			if(!std::strcmp(str, "uri"))
				state=State::TRPPREDURI;
			else if(!std::strcmp(str, "annotator"))
				state=State::TRPPREDANN;
			return true;
		case State::TRPSUB:
			if(!std::strcmp(str, "uri"))
				state=State::TRPSUBURI;
			else if(!std::strcmp(str, "boundaries"))
				state=State::TRPSUBBFIRST;
			else if(!std::strcmp(str, "annotator"))
				state=State::TRPSUBANN;
			return true;
		case State::TRPOBJ:
			if(!std::strcmp(str, "uri"))
				state=State::TRPOBJURI;
			else if(!std::strcmp(str, "boundaries"))
				state=State::TRPOBJBFIRST;
			else if(!std::strcmp(str, "annotator"))
				state=State::TRPOBJANN;
			return true;
	}
	return true;
}

bool Handler::String(const char *str, SizeType size, bool){
	switch(state){
		case State::DOCID:
			docid=atoi(str+sizeof("http://www.wikidata.org/entity/Q")-1);
			state=State::DOC;
			return true;
		case State::TEXT:
			memcpy(text, str, size);
			text_size=size;
			state=State::DOC;
			return true;
		case State::TRPPREDURI:
			triples.back().predicate=atoi(str+sizeof("http://www.wikidata.org/prop/direct/P")-1);
			state=State::TRPPRED;
			return true;
		case State::TRPANN:
			if(spoonly && std::strcmp(str, "SPOAligner")
					|| !spoonly && !std::strcmp(str, "NoSubject-Triple-aligner"))
				ignore_triple=true;
			state=State::TRP;
			return true;
		case State::TRPSUBURI:
			triples.back().subject=atoi(str+sizeof("http://www.wikidata.org/entity/Q")-1);
			state=State::TRPSUB;
			return true;
		case State::TRPOBJURI:
			triples.back().object=atoi(str+sizeof("http://www.wikidata.org/entity/Q")-1);
			state=State::TRPOBJ;
			return true;
		case State::TRPPREDANN:
			if(spoonly && std::strcmp(str, "Wikidata_Property_Linker")
					|| !spoonly && !std::strcmp(str, "NoSubject-Triple-aligner"))
				ignore_triple=true;
			state=State::TRPPRED;
			return true;
		case State::TRPSUBANN:
			if(std::strcmp(str, "Wikidata_Spotlight_Entity_Linker"))
				ignore_triple=true;
			state=State::TRPSUB;
			return true;
		case State::TRPOBJANN:
			if(std::strcmp(str, "Wikidata_Spotlight_Entity_Linker"))
				ignore_triple=true;
			state=State::TRPOBJ;
			return true;
	}
	return true;

}

bool Handler::Uint(unsigned x){
	switch(state){
		case State::WBFIRST:
			// words_boundaries.push_back({x, 0});
			words_boundaries.push_back(std::make_tuple(x, 0));
			// words_boundaries.push_back({{x, 0}});
			state=State::WBSECOND;
			return true;
		case State::WBSECOND:
			std::get<1>(words_boundaries.back())=x;
			state=State::WBEND;
			return true;
		case State::SBFIRST:
			// sentences_boundaries.push_back({x, 0});
			sentences_boundaries.push_back(std::make_tuple(x, 0));
			// sentences_boundaries.push_back({{x, 0}});
			state=State::SBSECOND;
			return true;
		case State::SBSECOND:
			std::get<1>(sentences_boundaries.back())=x;
			state=State::SBEND;
			return true;
		case State::TRPSENID:
			triples.back().sentence=x;
			state=State::TRP;
			return true;
		case State::TRPSUBBFIRST:
			std::get<0>(triples.back().subject_boundaries)=x;
			state=State::TRPSUBBSECOND;
			return true;
		case State::TRPSUBBSECOND:
			std::get<1>(triples.back().subject_boundaries)=x;
			state=State::TRPSUB;
			return true;
		case State::TRPOBJBFIRST:
			std::get<0>(triples.back().object_boundaries)=x;
			state=State::TRPOBJBSECOND;
			return true;
		case State::TRPOBJBSECOND:
			std::get<1>(triples.back().object_boundaries)=x;
			state=State::TRPOBJ;
			return true;
	}
	return true;
}

bool Handler::Null(){
	switch(state){
		case State::TRPSUBBFIRST:
			ignore_triple = true;
			state=State::TRPSUB;
			return true;
		case State::TRPOBJBFIRST:
			ignore_triple = true;
			state=State::TRPOBJ;
			return true;
	}
	return true;
}

void Handler::write_doc(){
	++nb_doc;
	if(!triples.size())
		return;
	nb_triple += triples.size();

	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cnv;
	std::u32string u32text = cnv.from_bytes(text, text+text_size);

	for(auto trp: triples){
		std::size_t b, e, b1, e1, b2, e2;
		std::tie(b , e )=sentences_boundaries[trp.sentence];
		std::tie(b1, e1)=trp.subject_boundaries;
		std::tie(b2, e2)=trp.object_boundaries;
		char const *reverse="";

		if(b1>b2){
			reverse="R";
			std::swap(b1, b2);
			std::swap(e1, e2);
		}

		if(b2<=e1){
			--nb_triple;
			continue;
		}

		bool pe1=false, pe2=false;
		for(auto it=std::lower_bound(begin(words_boundaries), end(words_boundaries), b,
					[](std::tuple<std::size_t,std::size_t> x, std::size_t n){ return std::get<0>(x)<n; });
				std::get<0>(*it)<e && it<end(words_boundaries); ++it){
			std::size_t wb, we;
			std::tie(wb, we) = *it;

			while(wb<we){
				std::size_t nwb=we, nwe=we;

				if(e1==wb && !pe1) // fix incorrect ".303 " entities
					out << "</e1> ";
				else if(e2==wb && !pe2)
					out << "</e2> ";
				else if(wb!=b)
					out << ' ';

				if(wb==b1)
					out << "<e1 q=\"" << trp.subject << "\">";
				else if(wb==b2)
					out << "<e2 q=\"" << trp.object << "\">";

				if(b1>wb && b1<we)
					we=nwb=b1;
				else if(e1>wb && e1<we)
					we=nwb=e1;
				else if(b2>wb && b2<we)
					we=nwb=b2;
				else if(e2>wb && e2<we)
					we=nwb=e2;

				out << cnv.to_bytes(u32text.substr(wb, we-wb));
				if(we==e1){
					out << "</e1>";
					pe1=true;
				} else if(we==e2){
					out << "</e2>";
					pe2=true;
				}

				wb=nwb;
				we=nwe;
			}
		}

		out << '\t' << docid << '\t' << reverse << trp.predicate << '\n';
	}
}

int main(int argc, char **argv){
	if(argc<3){
		std::cerr << "usage: " << argv[0] << " [-spo] outfile infiles...";
		return 1;
	}

	bool spoonly=!std::strcmp(argv[1], "-spo");
	std::vector<std::string> inpaths(argv+2+spoonly, argv+argc);

	std::ofstream outfile(argv[1+spoonly]);
	Handler handler(outfile, spoonly);
	Reader reader;

	char buffer[1<<16];
	std::size_t i=0;
	for(std::string inpath: inpaths){
		std::FILE *infile=fopen(inpath.c_str(), "rb");
		FileReadStream instream(infile, buffer, sizeof(buffer));
		reader.Parse(instream, handler);
		fclose(infile);
		std::clog << '\r' << ++i << '/' << inpaths.size() << ' '
			<< "DOC: " << handler.nb_doc << ' '
			<< "TRIPLE: " << handler.nb_triple << ' '
			<< "current: " << inpath << std::flush;
	}
	std::clog << '\n';
}
